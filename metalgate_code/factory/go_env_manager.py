"""Guest-compatible Go build environment management for microsandbox.

The official ``golang`` image includes the Go toolchain, gcc, git, and
make, but lacks development tools like ``gopls``, ``goimports``, and
``staticcheck``.  We install these into the image's default ``/go/bin``
on every sandbox boot.  Nothing is persisted across launches.

``GOTOOLCHAIN=auto`` lets Go auto-download the toolchain version required
by ``go.mod``.  We explicitly set ``GOPATH`` and ``GOCACHE`` to VM-local
paths so the host's values (leaked via ``inherit_env=True``) don't cause
I/O errors.

Setup runs only when the bind-mounted project is actually a Go module
(i.e. a ``go.mod`` exists in the workdir).  A Go-capable image alone is
not enough — see :func:`is_go_image` for the cheap pre-filter used by
the backend.
"""

from __future__ import annotations

import logging

from microsandbox import Sandbox

logger = logging.getLogger("metalgate_code")

SANDBOX_WORKDIR = "/workspace"

_GO_SETUP_TIMEOUT_SEC = 300
"""Timeout for the Go dev tool install step."""

_GO_IMAGE_MARKERS = ("golang", "go:")
"""Substrings that identify a Go-capable image (for env setup)."""

_GO_IMAGE_BIN = "/usr/local/go/bin"
"""Where the Go toolchain lives in the official ``golang`` image."""

_GO_DEFAULT_GOPATH = "/go"
"""The image's default ``GOPATH`` (on the ephemeral VM filesystem)."""

_GO_DEFAULT_GOCACHE = "/root/.cache/go-build"
"""VM-local build cache (avoids host path leaking via inherit_env)."""

_GOTOOLCHAIN = "auto"
"""Let Go auto-download the toolchain version required by ``go.mod``."""

# Go development tools installed via ``go install``.  Each tool is installed
# in its own ``go install`` call: ``gopls@latest`` and ``goimports@latest``
# resolve to different versions of ``golang.org/x/tools``, and ``go install``
# with multiple packages requires a single module version.
_GO_DEV_TOOLS = (
    "golang.org/x/tools/gopls@latest",
    "golang.org/x/tools/cmd/goimports@latest",
    "golang.org/x/tools/cmd/gorename@latest",
    "honnef.co/go/tools/cmd/staticcheck@latest",
    "golang.org/x/vuln/cmd/govulncheck@latest",
    "github.com/go-delve/delve/cmd/dlv@latest",
)


def is_go_image(image: str) -> bool:
    """Whether the image is Go-capable (triggers Go env setup)."""
    img = image.lower()
    return any(marker in img for marker in _GO_IMAGE_MARKERS)


def build_go_env(base_path: str) -> dict[str, str]:
    """Build the env dict that activates a Go env for ``sb.shell(env=...)``.

    Uses the image's defaults (``GOPATH=/go``, ``GOBIN=/go/bin``).
    ``GOTOOLCHAIN=auto`` lets Go auto-download the toolchain version
    required by ``go.mod``.  ``GOCACHE`` and ``TMPDIR`` are set to VM-local
    paths so the host's values (leaked via ``inherit_env=True``) don't cause
    I/O errors — Go uses ``TMPDIR`` for compilation work dirs.

    ``base_path`` is the VM's real ``$PATH`` (captured during env setup),
    since env dict values are not shell-expanded.
    """
    return {
        "GOPATH": _GO_DEFAULT_GOPATH,
        "GOCACHE": _GO_DEFAULT_GOCACHE,
        "PATH": f"{_GO_IMAGE_BIN}:/go/bin:{base_path}",
        "GOENV": "off",
        "GOTOOLCHAIN": _GOTOOLCHAIN,
        "TMPDIR": "/tmp",
    }


class GoEnvManager:
    """Installs Go dev tools into a microsandbox VM on each boot.

    Depends on a single VM primitive (run command) provided by the
    backend, so it stays decoupled from the sandbox lifecycle.
    """

    def __init__(self, sb: Sandbox, *, run_in_vm) -> None:
        self._sb = sb
        self._run_in_vm = run_in_vm

    async def ensure(self) -> dict[str, str] | None:
        """Ensure a Go build env is active; return the env dict, or ``None``.

        Returns ``None`` when the bind-mounted project is not a Go module
        (no ``go.mod`` in the workdir) or when the Go toolchain is absent.
        Dev-tool installation is best-effort: a failed tool does not
        prevent the env from being returned, since ``GOTOOLCHAIN=auto``
        and the PATH are still useful for plain ``go build``/``go test``.
        """
        # Only run for actual Go projects.
        if not await self._has_go_module():
            logger.info("No go.mod in %s; skipping Go env setup", SANDBOX_WORKDIR)
            return None

        # Relink /bin/sh to bash so the execute tool (which invokes
        # /bin/sh) gets bash semantics instead of dash.  The official
        # golang Debian image ships bash at /usr/bin/bash but leaves
        # /bin/sh pointing to dash.
        await self._relink_shell_to_bash()

        # Verify the Go toolchain is present.  We call go by its absolute
        # path because inherit_env=True leaks the host's PATH into the VM,
        # which doesn't include /usr/local/go/bin.
        go_res = await self._run_in_vm(
            self._sb, f"{_GO_IMAGE_BIN}/go version", timeout=15, env=None
        )
        if go_res.exit_code != 0:
            logger.warning(
                "go not found in image (exit %s): %s",
                go_res.exit_code,
                go_res.output[-1000:],
            )
            return None

        # Capture the VM's real PATH.  Same reason as above: env=None gives
        # us the host-leaked PATH, but we need the image's default PATH so
        # build_go_env produces a useful PATH for subsequent commands.
        # Run a login shell to get the image's default PATH.
        path_res = await self._run_in_vm(
            self._sb,
            'sh -l -c "echo $PATH"',
            timeout=10,
            env=None,
        )
        vm_path = path_res.output.strip() if path_res.exit_code == 0 else ""

        # Install Go dev tools (best-effort).
        await self._install_dev_tools(vm_path)

        return build_go_env(vm_path)

    async def _has_go_module(self) -> bool:
        """Whether a ``go.mod`` exists in the VM workdir."""
        res = await self._run_in_vm(
            self._sb,
            f"test -f {SANDBOX_WORKDIR}/go.mod",
            timeout=10,
            env=None,
        )
        return res.exit_code == 0

    async def _relink_shell_to_bash(self) -> None:
        """Relink ``/bin/sh`` to bash so the execute tool gets bash semantics.

        The official ``golang`` Debian image ships bash at
        ``/usr/bin/bash`` but leaves ``/bin/sh`` pointing to dash.  The
        execute tool invokes ``/bin/sh``, so the agent is stuck with
        POSIX-only dash semantics.  This relinks ``/bin/sh`` → bash on
        each boot (nothing is persisted across launches).

        Best-effort: a failure is logged but does not abort setup.
        """
        res = await self._run_in_vm(
            self._sb,
            "ln -sf /usr/bin/bash /bin/sh",
            timeout=10,
            env=None,
        )
        if res.exit_code != 0:
            logger.warning(
                "Failed to relink /bin/sh to bash (exit %s): %s",
                res.exit_code,
                res.output[-500:],
            )
        else:
            logger.info("Relinked /bin/sh -> /usr/bin/bash")

    async def _install_dev_tools(self, vm_path: str) -> None:
        """Install Go dev tools (gopls, goimports, staticcheck) into ``/go/bin``.

        Each tool is installed in its own ``go install`` call.  Installs run
        from ``/tmp`` (not the project root) so the image's bundled Go is used
        for the tool build, not the toolchain version pinned by the project's
        ``go.mod`` — tool installation should not trigger a per-project
        toolchain switch.

        Failures are logged but do not abort: ``gopls`` is the only tool the
        agent depends on, and even its absence leaves a working build env.
        """
        env = build_go_env(vm_path)
        for tool in _GO_DEV_TOOLS:
            cmd = f"cd /tmp && go install {tool}"
            res = await self._run_in_vm(
                self._sb,
                cmd,
                timeout=_GO_SETUP_TIMEOUT_SEC,
                env=env,
            )
            if res.exit_code != 0:
                logger.warning(
                    "go install %s failed (exit %s): %s",
                    tool,
                    res.exit_code,
                    res.output[-2000:],
                )
            else:
                logger.info("Installed %s", tool)
