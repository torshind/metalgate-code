"""Unit tests for Go contextual symbol search tools.

These tests use ``LocalShellBackend`` and run on the host without a
sandbox.  Sandbox/agent integration tests live in
``test_go_context_integration.py``.

The ``TestGotoDefinition`` class is the critical regression suite.  It
exercises the exact failure modes that were reported in
``goto_definition_bugfix_report.md`` and verified against a real gopls
v0.23.0 instance.  Every assertion encodes a *ground-truth* gopls
response, so a failure pinpoints the exact bug.

Ground truth (captured from `gopls definition -json` at the *member*
column of each call site in orders.go):

    orders.go line 40:  return strings.ToUpper(o.Process())
        strings.ToUpper @ col 17 -> strings.go:687   func strings.ToUpper(s string) string
        o.Process      @ col 27 -> orders.go:25      func (o *Order) Process() string

    orders.go line 26:  if !ValidateAddress(o.Address) {
        ValidateAddress @ col 6  -> validation.go:7   func ValidateAddress(address string) bool
        o.Address       @ col 24 -> orders.go:7        field Address string

    orders.go line 29:  formatted := FormatCurrency(o.Amount)
        FormatCurrency @ col 15 -> utils.go:6         func FormatCurrency(amount float64) string
        o.Amount       @ col 32 -> orders.go:8        field Amount float64

Key insight: for a selector expression ``pkg.Func`` / ``recv.Method`` /
``obj.Field``, gopls must be queried at the column of the **member**
(the part after the ``.``), NOT the qualifier/receiver or the dot.  If
the column points at the qualifier or the dot, gopls resolves to the
package import, the variable declaration, or the receiver type instead
of the target symbol.
"""

import tempfile
from pathlib import Path

import pytest
from deepagents.backends import LocalShellBackend

from metalgate_code.context import get_code_tools

SAMPLE_DIR = Path(__file__).parent / "sample" / "go" / "simple"
ORDERS_FILE = str(SAMPLE_DIR / "orders.go")
VALIDATION_FILE = str(SAMPLE_DIR / "validation.go")
UTILS_FILE = str(SAMPLE_DIR / "utils.go")
PROCESSOR_FILE = str(SAMPLE_DIR / "processor.go")
MULTISELECTOR_FILE = str(SAMPLE_DIR / "multiselector.go")
EXTERNAL_FILE = str(SAMPLE_DIR / "external.go")


@pytest.fixture(scope="module")
def tools():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    shell_backend = LocalShellBackend(
        root_dir=str(SAMPLE_DIR),
        virtual_mode=False,
        inherit_env=True,
    )

    tool_list = get_code_tools(
        cwd=str(SAMPLE_DIR),
        backend=shell_backend,
        cache_path=db_path,
        language="go",
    )
    (
        goto_def,
        outline,
        get_source,
        callers,
        callees,
        find_sym,
    ) = tool_list

    yield {
        "goto_definition": goto_def,
        "get_file_outline": outline,
        "get_source": get_source,
        "get_callers": callers,
        "get_callees": callees,
        "find_symbol": find_sym,
    }


# get_file_outline
class TestGetFileOutline:
    def test_finds_struct(self, tools):
        symbols = tools["get_file_outline"](ORDERS_FILE)
        assert any(s["name"] == "Order" and s["kind"] == "struct" for s in symbols)

    def test_finds_interface(self, tools):
        symbols = tools["get_file_outline"](ORDERS_FILE)
        assert any(
            s["name"] == "Processor" and s["kind"] == "interface" for s in symbols
        )

    def test_finds_function(self, tools):
        symbols = tools["get_file_outline"](ORDERS_FILE)
        assert any(s["name"] == "NewOrder" and s["kind"] == "function" for s in symbols)

    def test_finds_method(self, tools):
        symbols = tools["get_file_outline"](ORDERS_FILE)
        method = next(
            (s for s in symbols if s["name"] == "Process" and s["kind"] == "method"),
            None,
        )
        assert method is not None

    def test_method_has_receiver(self, tools):
        symbols = tools["get_file_outline"](ORDERS_FILE)
        process = next(s for s in symbols if s["name"] == "Process")
        assert "Order" in (process.get("class") or "")

    def test_signature_contains_name(self, tools):
        symbols = tools["get_file_outline"](VALIDATION_FILE)
        validate = next(s for s in symbols if s["name"] == "ValidateAddress")
        assert "ValidateAddress" in validate["signature"]

    def test_cached_result_is_identical(self, tools):
        first = tools["get_file_outline"](ORDERS_FILE)
        second = tools["get_file_outline"](ORDERS_FILE)
        assert first == second


# get_source
class TestGetSource:
    def _validate_line(self, tools):
        symbols = tools["get_file_outline"](VALIDATION_FILE)
        return next(s for s in symbols if s["name"] == "ValidateAddress")["line"]

    def test_source_contains_func(self, tools):
        line = self._validate_line(tools)
        result = tools["get_source"](VALIDATION_FILE, line)
        assert "ValidateAddress" in result["source"]
        assert "return false" in result["source"]

    def test_start_and_end_lines_are_sane(self, tools):
        line = self._validate_line(tools)
        result = tools["get_source"](VALIDATION_FILE, line)
        assert result["start_line"] >= 1
        assert result["end_line"] >= result["start_line"]

    def test_get_source_for_struct(self, tools):
        symbols = tools["get_file_outline"](ORDERS_FILE)
        st = next(s for s in symbols if s["name"] == "Order")
        result = tools["get_source"](ORDERS_FILE, st["line"])
        assert "type Order struct" in result["source"]

    def test_get_source_from_body_line(self, tools):
        """get_source should work when given any line inside the function body."""
        symbols = tools["get_file_outline"](VALIDATION_FILE)
        va = next(s for s in symbols if s["name"] == "ValidateAddress")
        body_line = va["line"] + 2  # inside the body
        result = tools["get_source"](VALIDATION_FILE, body_line)
        assert "ValidateAddress" in result["source"]

    def test_fallback_context_window(self, tools):
        result = tools["get_source"](VALIDATION_FILE, 1, context=10)
        assert isinstance(result["source"], str)

    def test_nonexistent_file_returns_error(self, tools):
        result = tools["get_source"]("/nonexistent/file.go", 1)
        assert result["source"] == "" or "error" in result


# find_symbol
class TestFindSymbol:
    def test_exact_match_finds_validate_address(self, tools):
        results = tools["find_symbol"]("ValidateAddress")
        names = [r["name"] for r in results]
        assert "ValidateAddress" in names

    def test_exact_match_does_not_find_partial(self, tools):
        results = tools["find_symbol"]("Validate")
        names = [r["name"] for r in results]
        assert "ValidateAddress" not in names

    def test_case_insensitive(self, tools):
        results = tools["find_symbol"]("validateaddress")
        names = [r["name"] for r in results]
        assert "ValidateAddress" in names

    def test_unknown_symbol_returns_empty_list(self, tools):
        results = tools["find_symbol"]("zzz_does_not_exist_xyz")
        assert results == []

    def test_finds_struct_by_name(self, tools):
        results = tools["find_symbol"]("Order")
        names = [r["name"] for r in results]
        assert "Order" in names

    def test_cached_result_is_identical(self, tools):
        first = tools["find_symbol"]("ValidateAddress")
        second = tools["find_symbol"]("ValidateAddress")
        assert first == second

    def test_results_scoped_to_project_root(self, tools):
        """find_symbol should only return symbols within the project root,
        not from other projects, stdlib, or the module cache."""
        results = tools["find_symbol"]("Order")
        for r in results:
            assert str(SAMPLE_DIR) in r["file"], (
                f"Result file {r['file']} is outside project root {SAMPLE_DIR}"
            )


# ---------------------------------------------------------------------------
# goto_definition
# ---------------------------------------------------------------------------
#
# This is the critical regression suite.  Each test resolves a specific
# call site in orders.go and asserts the EXACT location/kind/signature that
# a real gopls returns when queried at the member column.
#
# The call sites (all in orders.go):
#
#   line 40:  return strings.ToUpper(o.Process())
#   line 26:  if !ValidateAddress(o.Address) {
#   line 29:  formatted := FormatCurrency(o.Amount)
#
# Ground truth captured from `gopls definition -json` at the member column:
#
#   strings.ToUpper  -> strings.go:687  "func strings.ToUpper(s string) string"
#   o.Process        -> orders.go:25    "func (o *Order) Process() string"
#   ValidateAddress  -> validation.go:7  "func ValidateAddress(address string) bool"
#   o.Address        -> orders.go:7      "field Address string"
#   FormatCurrency   -> utils.go:6       "func FormatCurrency(amount float64) string"
#   o.Amount         -> orders.go:8      "field Amount float64"
#
# The bug under test: for selector expressions (pkg.Func, recv.Method,
# obj.Field) the tool sends gopls the column of the qualifier/receiver or
# the dot instead of the member.  gopls then resolves to the import line,
# the variable declaration, or the receiver type — NOT the target symbol.
# Every test below fails when that bug is present and passes once the
# column computation is fixed to point at the member.
# ---------------------------------------------------------------------------


class TestGotoDefinition:
    """Resolves call sites in orders.go and checks against gopls ground truth.

    All call sites live in orders.go lines 26, 29, and 40:

        26:  if !ValidateAddress(o.Address) {
        29:  formatted := FormatCurrency(o.Amount)
        40:  return strings.ToUpper(o.Process())
    """

    # ---- helpers ---------------------------------------------------------

    @staticmethod
    def _basename(path: str) -> str:
        return path.replace("\\", "/").rstrip("/").split("/")[-1]

    # ---- stdlib qualified call: strings.ToUpper --------------------------

    def test_resolves_qualified_stdlib_call(self, tools):
        """strings.ToUpper (orders.go:40) must resolve to the stdlib function,
        not the `import "strings"` line.

        gopls ground truth at the member column (col 17, the 'T' of ToUpper):
            strings.go:687  func strings.ToUpper(s string) string
        """
        result = tools["goto_definition"](ORDERS_FILE, 40, "strings.ToUpper")
        assert self._basename(result["file"]) == "strings.go", (
            f"expected strings.go, got {result['file']} "
            "(resolving the package import instead of the function)"
        )
        assert result["line"] == 687, f"expected line 687, got {result['line']}"
        assert "ToUpper" in result["signature"], (
            f"expected signature containing 'ToUpper', got {result['signature']!r}"
        )

    def test_qualified_stdlib_call_kind_is_function(self, tools):
        """strings.ToUpper must be classified as a function, not 'unknown'."""
        result = tools["goto_definition"](ORDERS_FILE, 40, "strings.ToUpper")
        assert result["kind"] == "function", (
            f"expected kind 'function', got {result['kind']!r} "
            "(kind inference is falling back to 'unknown' because the hover "
            "signature is the package docstring, not the function signature)"
        )

    def test_qualified_stdlib_call_not_import_line(self, tools):
        """The result must NOT be the `import "strings"` line in orders.go.

        This is a regression guard for the column bug: when the column points
        at the qualifier `strings` (or the dot), gopls returns the import
        statement with signature 'package strings'.  The result file would be
        orders.go and the line would be 3 (the import line).
        """
        result = tools["goto_definition"](ORDERS_FILE, 40, "strings.ToUpper")
        assert not (
            self._basename(result["file"]) == "orders.go" and result["line"] == 3
        ), (
            "resolved to the import line — column points at the qualifier, "
            "not the member"
        )

    # ---- concrete method call: o.Process ---------------------------------

    def test_resolves_method_call_on_receiver(self, tools):
        """o.Process (orders.go:40) must resolve to the method definition,
        not the `var o *Order` declaration or the receiver type.

        gopls ground truth at the member column (col 27, the 'P' of Process):
            orders.go:25  func (o *Order) Process() string
        """
        result = tools["goto_definition"](ORDERS_FILE, 40, "o.Process")
        assert self._basename(result["file"]) == "orders.go", (
            f"expected orders.go, got {result['file']}"
        )
        assert result["line"] == 25, f"expected line 25, got {result['line']}"
        assert "Process" in result["signature"], (
            f"expected signature containing 'Process', got {result['signature']!r}"
        )

    def test_method_call_kind_is_method(self, tools):
        """o.Process must be classified as a method, not 'unknown'."""
        result = tools["goto_definition"](ORDERS_FILE, 40, "o.Process")
        assert result["kind"] == "method", (
            f"expected kind 'method', got {result['kind']!r}"
        )

    def test_method_call_not_receiver_declaration(self, tools):
        """The result must NOT be the receiver variable declaration.

        Regression guard: when the column points at the receiver `o` (or the
        dot), gopls returns `var o *Order` at the function signature line
        (line 25 in older gopls, or the parameter line).  The signature would
        be 'var o *Order' instead of the method signature.
        """
        result = tools["goto_definition"](ORDERS_FILE, 40, "o.Process")
        assert "var " not in result["signature"], (
            f"resolved to a variable declaration: {result['signature']!r} "
            "(column points at the receiver, not the member)"
        )

    # ---- plain function call: ValidateAddress ----------------------------

    def test_resolves_plain_function_call(self, tools):
        """ValidateAddress (orders.go:26) must resolve to validation.go.

        gopls ground truth at col 6:
            validation.go:7  func ValidateAddress(address string) bool
        """
        result = tools["goto_definition"](ORDERS_FILE, 26, "ValidateAddress")
        assert self._basename(result["file"]) == "validation.go", (
            f"expected validation.go, got {result['file']}"
        )
        assert result["line"] == 7, f"expected line 7, got {result['line']}"
        assert "ValidateAddress" in result["signature"]

    def test_plain_function_call_kind_is_function(self, tools):
        result = tools["goto_definition"](ORDERS_FILE, 26, "ValidateAddress")
        assert result["kind"] == "function", (
            f"expected kind 'function', got {result['kind']!r}"
        )

    # ---- plain function call: FormatCurrency -----------------------------

    def test_resolves_plain_function_call_other_file(self, tools):
        """FormatCurrency (orders.go:29) must resolve to utils.go.

        gopls ground truth at col 15:
            utils.go:6  func FormatCurrency(amount float64) string
        """
        result = tools["goto_definition"](ORDERS_FILE, 29, "FormatCurrency")
        assert self._basename(result["file"]) == "utils.go", (
            f"expected utils.go, got {result['file']}"
        )
        assert result["line"] == 6, f"expected line 6, got {result['line']}"
        assert "FormatCurrency" in result["signature"]

    def test_plain_function_call_other_file_kind_is_function(self, tools):
        result = tools["goto_definition"](ORDERS_FILE, 29, "FormatCurrency")
        assert result["kind"] == "function", (
            f"expected kind 'function', got {result['kind']!r}"
        )

    # ---- struct field access: o.Address ----------------------------------

    def test_resolves_struct_field_access(self, tools):
        """o.Address (orders.go:26) must resolve to the struct field,
        not the `var o *Order` declaration.

        gopls ground truth at the member column (col 24, the 'A' of Address):
            orders.go:7  field Address string
        """
        result = tools["goto_definition"](ORDERS_FILE, 26, "o.Address")
        assert self._basename(result["file"]) == "orders.go", (
            f"expected orders.go, got {result['file']}"
        )
        assert result["line"] == 7, f"expected line 7, got {result['line']}"
        assert "Address" in result["signature"], (
            f"expected signature containing 'Address', got {result['signature']!r}"
        )

    def test_struct_field_access_not_receiver_declaration(self, tools):
        """o.Address must NOT resolve to `var o *Order`.

        Regression guard: when the column points at the receiver `o`, gopls
        returns the variable declaration `var o *Order` at line 25.
        """
        result = tools["goto_definition"](ORDERS_FILE, 26, "o.Address")
        assert "var " not in result["signature"], (
            f"resolved to a variable declaration: {result['signature']!r} "
            "(column points at the receiver, not the member)"
        )

    # ---- struct field access: o.Amount -----------------------------------

    def test_resolves_struct_field_access_amount(self, tools):
        """o.Amount (orders.go:29) must resolve to the struct field.

        gopls ground truth at the member column (col 32, the 'A' of Amount):
            orders.go:8  field Amount float64
        """
        result = tools["goto_definition"](ORDERS_FILE, 29, "o.Amount")
        assert self._basename(result["file"]) == "orders.go", (
            f"expected orders.go, got {result['file']}"
        )
        assert result["line"] == 8, f"expected line 8, got {result['line']}"
        assert "Amount" in result["signature"], (
            f"expected signature containing 'Amount', got {result['signature']!r}"
        )

    def test_struct_field_access_amount_not_receiver_declaration(self, tools):
        result = tools["goto_definition"](ORDERS_FILE, 29, "o.Amount")
        assert "var " not in result["signature"], (
            f"resolved to a variable declaration: {result['signature']!r} "
            "(column points at the receiver, not the member)"
        )

    # ---- interface method call: p.Process --------------------------------
    #
    # processor.go defines UseProcessor(p Processor) which calls p.Process().
    # gopls ground truth at the member column (col 11, the 'P' of Process):
    #     orders.go:13  func (Processor) Process() string
    # (the interface method line inside `type Processor interface`)

    def test_resolves_interface_method_call(self, tools):
        """p.Process (where p is of interface type Processor) must resolve
        to the method, not the interface type or the parameter declaration.

        gopls ground truth at the member column:
            orders.go:13  func (Processor) Process() string
        (the interface method line inside `type Processor interface`)

        Note: some gopls versions resolve interface-method calls to the
        `type Processor interface` line (line 12) rather than the method
        line (line 13).  Both are acceptable as long as the result is NOT
        the parameter declaration `var p Processor` and the kind is
        'method'.
        """
        result = tools["goto_definition"](PROCESSOR_FILE, 5, "p.Process")
        # Must resolve into orders.go (where the interface is defined),
        # not back to the parameter declaration in processor.go.
        assert self._basename(result["file"]) == "orders.go", (
            f"expected orders.go, got {result['file']} "
            "(resolving the parameter declaration instead of the method)"
        )
        assert result["line"] in (12, 13), (
            f"expected line 12 or 13, got {result['line']}"
        )
        assert result["kind"] == "method", (
            f"expected kind 'method', got {result['kind']!r}"
        )

    def test_interface_method_call_not_parameter_declaration(self, tools):
        """p.Process must NOT resolve to `var p Processor`."""
        result = tools["goto_definition"](PROCESSOR_FILE, 5, "p.Process")
        assert "var " not in result["signature"], (
            f"resolved to a variable declaration: {result['signature']!r} "
            "(column points at the receiver, not the member)"
        )

    # ---- same-line disambiguation ----------------------------------------
    #
    # orders.go:40 has TWO selectors on the same line:
    #     return strings.ToUpper(o.Process())
    # A correct implementation must resolve each one independently based on
    # the name argument, not blindly pick the first selector on the line.

    def test_same_line_disambiguation_first_selector(self, tools):
        """On line 40, `strings.ToUpper` must resolve to strings.go (the
        first selector), not orders.go (the second selector's method)."""
        result = tools["goto_definition"](ORDERS_FILE, 40, "strings.ToUpper")
        assert self._basename(result["file"]) == "strings.go", (
            f"expected strings.go, got {result['file']}"
        )

    def test_same_line_disambiguation_second_selector(self, tools):
        """On line 40, `o.Process` must resolve to orders.go:25 (the second
        selector), not strings.go (the first selector)."""
        result = tools["goto_definition"](ORDERS_FILE, 40, "o.Process")
        assert self._basename(result["file"]) == "orders.go", (
            f"expected orders.go, got {result['file']}"
        )
        assert result["line"] == 25, f"expected line 25, got {result['line']}"

    # ---- result shape ----------------------------------------------------

    def test_result_has_required_fields(self, tools):
        """Every result must include name, kind, file, line, col, signature."""
        result = tools["goto_definition"](ORDERS_FILE, 26, "ValidateAddress")
        for field in ("name", "kind", "file", "line", "col", "signature"):
            assert field in result, f"missing field {field!r} in result"

    def test_col_is_member_column_for_selector(self, tools):
        """For a selector expression, the returned `col` must point at the
        member (the part after the dot), not the qualifier or the dot.

        For strings.ToUpper on line 40, the member 'ToUpper' starts at
        column 17 (1-indexed).  The returned col must be >= 17 and point
        within the 'ToUpper' token, i.e. in the range [17, 23].
        """
        result = tools["goto_definition"](ORDERS_FILE, 40, "strings.ToUpper")
        assert 17 <= result["col"] <= 23, (
            f"expected col in [17, 23] (the 'ToUpper' member), "
            f"got col={result['col']} "
            "(column points at the qualifier or dot, not the member)"
        )

    def test_col_is_member_column_for_method(self, tools):
        """For o.Process on line 40, the member 'Process' starts at column 27.
        The returned col must be in [27, 34]."""
        result = tools["goto_definition"](ORDERS_FILE, 40, "o.Process")
        assert 27 <= result["col"] <= 34, (
            f"expected col in [27, 34] (the 'Process' member), got col={result['col']}"
        )

    def test_col_is_member_column_for_field(self, tools):
        """For o.Address on line 26, the member 'Address' starts at column 24.
        The returned col must be in [24, 31]."""
        result = tools["goto_definition"](ORDERS_FILE, 26, "o.Address")
        assert 24 <= result["col"] <= 31, (
            f"expected col in [24, 31] (the 'Address' member), got col={result['col']}"
        )


# get_goto_definition — same qualifier appearing twice on one line
#
# This is the pattern that the gin codebase exercises but the original test
# suite missed.  In gin's recovery.go:56:
#
#     logger = log.New(out, "\n\n\x1b[31m", log.LstdFlags)
#
# the qualifier `log` appears twice on the same line (log.New and
# log.LstdFlags).  A column computation that finds the *first* occurrence of
# the qualifier, or the *first* dot on the line, will resolve the wrong
# symbol.  The tool must use the `name` argument to find the exact
# `qualifier.member` pair and compute the column of *that* member.
#
# multiselector.go:8 has the same shape with `strings` as the qualifier:
#
#     return strings.ToUpper(strings.ToLower(s))
#
# gopls ground truth (captured from gopls v0.23.0):
#     strings.ToUpper @ col 17 -> strings.go:687  func strings.ToUpper(s string) string
#     strings.ToLower @ col 33 -> strings.go:727  func strings.ToLower(s string) string
class TestGotoDefinitionSameQualifierTwice:
    """Resolves two selectors that share the same qualifier on one line.

    multiselector.go line 8:
        return strings.ToUpper(strings.ToLower(s))

    Both selectors use the qualifier `strings`.  The tool must distinguish
    them by the member name in the `name` argument, not by finding the
    first `strings.` on the line.
    """

    @staticmethod
    def _basename(path: str) -> str:
        return path.replace("\\", "/").rstrip("/").split("/")[-1]

    def test_first_occurrence_resolves_correctly(self, tools):
        """strings.ToUpper (the first `strings.` on line 8) must resolve to
        strings.go:687, not the import line."""
        result = tools["goto_definition"](MULTISELECTOR_FILE, 8, "strings.ToUpper")
        assert self._basename(result["file"]) == "strings.go", (
            f"expected strings.go, got {result['file']} "
            "(resolving the package import instead of the function)"
        )
        assert result["line"] == 687, f"expected line 687, got {result['line']}"
        assert "ToUpper" in result["signature"], (
            f"expected signature containing 'ToUpper', got {result['signature']!r}"
        )

    def test_second_occurrence_resolves_correctly(self, tools):
        """strings.ToLower (the SECOND `strings.` on line 8) must resolve to
        strings.go:727, not strings.go:687 (the first occurrence) and not
        the import line."""
        result = tools["goto_definition"](MULTISELECTOR_FILE, 8, "strings.ToLower")
        assert self._basename(result["file"]) == "strings.go", (
            f"expected strings.go, got {result['file']} "
            "(resolving the package import instead of the function)"
        )
        assert result["line"] == 727, (
            f"expected line 727 (ToLower), got {result['line']} "
            "(resolving the first occurrence ToUpper instead of the requested ToLower)"
        )
        assert "ToLower" in result["signature"], (
            f"expected signature containing 'ToLower', got {result['signature']!r}"
        )

    def test_second_occurrence_not_first(self, tools):
        """The result for strings.ToLower must NOT be the same as
        strings.ToUpper.  This catches a column computation that always
        picks the first selector on the line regardless of the name arg."""
        first = tools["goto_definition"](MULTISELECTOR_FILE, 8, "strings.ToUpper")
        second = tools["goto_definition"](MULTISELECTOR_FILE, 8, "strings.ToLower")
        assert first["line"] != second["line"], (
            f"both selectors resolved to the same line {first['line']} "
            "(column computation is picking the first selector, not the "
            "one matching the name argument)"
        )

    def test_second_occurrence_not_import_line(self, tools):
        """strings.ToLower must NOT resolve to the import line."""
        result = tools["goto_definition"](MULTISELECTOR_FILE, 8, "strings.ToLower")
        assert not (
            self._basename(result["file"]) == "multiselector.go" and result["line"] == 3
        ), (
            "resolved to the import line — column points at the qualifier, "
            "not the member"
        )

    def test_col_points_at_correct_member_first(self, tools):
        """For strings.ToUpper on line 8, col must be in [17, 24] (the
        'ToUpper' token), not at the qualifier or dot."""
        result = tools["goto_definition"](MULTISELECTOR_FILE, 8, "strings.ToUpper")
        assert 17 <= result["col"] <= 24, (
            f"expected col in [17, 24] (the 'ToUpper' member), got col={result['col']}"
        )

    def test_col_points_at_correct_member_second(self, tools):
        """For strings.ToLower on line 8, col must be in [33, 40] (the
        'ToLower' token), not at the first selector or the qualifier."""
        result = tools["goto_definition"](MULTISELECTOR_FILE, 8, "strings.ToLower")
        assert 33 <= result["col"] <= 40, (
            f"expected col in [33, 40] (the 'ToLower' member), "
            f"got col={result['col']} "
            "(column points at the first selector, not the requested one)"
        )


# 3rd-party package call
#
# external.go imports gin and calls gin.New().  This reproduces the exact
# bug reported in the gin codebase: a qualified call to a 3rd-party package
# resolves to the `import "github.com/gin-gonic/gin"` line (signature
# "package gin") instead of the actual function definition in the package.
#
# gopls ground truth (captured from gopls v0.23.0):
#     At the member column (col 13, the 'N' of New):
#         gin.go:202  func gin.New(opts ...gin.OptionFunc) *gin.Engine
#     At the dot (col 12):
#         external.go:3  package gin  (the import line — THIS IS THE BUG)
class TestGotoDefinitionThirdParty:
    """gin.New (external.go:7) must resolve to the function in the 3rd-party
    package, not the import line in the caller.

    external.go line 7:
        return gin.New()

    gopls ground truth at the member column:
        gin.go:202  func gin.New(opts ...gin.OptionFunc) *gin.Engine
    """

    @staticmethod
    def _basename(path: str) -> str:
        return path.replace("\\", "/").rstrip("/").split("/")[-1]

    def test_resolves_third_party_function(self, tools):
        """gin.New must resolve to gin.go (the 3rd-party package source),
        not external.go:3 (the import line in the caller)."""
        result = tools["goto_definition"](EXTERNAL_FILE, 7, "gin.New")
        assert result, "got empty result"
        assert self._basename(result["file"]) == "gin.go", (
            f"expected gin.go (the function definition), "
            f"got {result['file']} "
            "(resolving the package import instead of the function)"
        )
        assert "New" in result["signature"], (
            f"expected signature containing 'New', "
            f"got {result['signature']!r} "
            "(got the package docstring instead of the function signature)"
        )

    def test_third_party_not_import_line(self, tools):
        """The result must NOT be the import line in external.go.

        When the column points at the dot or qualifier, gopls returns the
        import statement: external.go:3 with signature 'package gin'.
        """
        result = tools["goto_definition"](EXTERNAL_FILE, 7, "gin.New")
        assert not (
            self._basename(result["file"]) == "external.go" and result["line"] == 3
        ), (
            "resolved to the import line — the column points at the "
            "qualifier/dot, not the member 'New'"
        )

    def test_third_party_kind_is_function(self, tools):
        """gin.New must be classified as a function, not 'unknown'."""
        result = tools["goto_definition"](EXTERNAL_FILE, 7, "gin.New")
        assert result["kind"] == "function", (
            f"expected kind 'function', got {result['kind']!r} "
            "(kind inference fell back to 'unknown' because the hover "
            "signature is the package docstring, not the function signature)"
        )
