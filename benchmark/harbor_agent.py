import tomllib
from pathlib import Path

from deepagents_acp.server import AgentSessionContext
from harbor import AgentContext, BaseAgent, BaseEnvironment

from benchmark.backend import HarborSandbox
from metalgate_code.factory.agent_factory import _build_agent


class HarborAgent(BaseAgent):
    def __init__(self, model_name: str = "evroc:moonshotai/Kimi-K2.5", **kwargs):
        self._model_name = model_name
        super().__init__(model_name=model_name, **kwargs)

    @staticmethod
    def name() -> str:
        return "metalgate"

    def version(self) -> str | None:
        return "0.1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        pyproject = tomllib.loads(Path("pyproject.toml").read_text())
        deps = pyproject["project"]["dependencies"]
        await environment.exec(f"pip install {' '.join(deps)}")

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        # HarborSandbox wraps Harbor's environment into DeepAgents' BackendProtocol
        backend = HarborSandbox(environment)

        cwd = (await environment.exec("pwd")).stdout
        if cwd:
            cwd = cwd.strip()
        await environment.upload_file(
            source_path=Path("benchmark/skills.py"),
            target_path=f"{cwd}/.metalgate/skills.py`",
        )

        session_context = AgentSessionContext(
            cwd=cwd if cwd else "/testbed",
            mode="accept_everything",
            model=self._model_name,
        )

        agent = _build_agent(session_context, backend)

        log_file = self.logs_dir / "agent.txt"

        with open(log_file, "w") as f:
            if cwd:
                f.write(f"cwd: {cwd}\n")
            else:
                f.write("Warning: pwd failed\n")
            async for chunk in agent.astream(
                {"messages": [{"role": "user", "content": instruction}]}
            ):
                line = str(chunk)
                f.write(line + "\n")
                f.flush()
