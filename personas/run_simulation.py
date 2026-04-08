"""
run_simulation.py — LitmusAI agent prompt generator.

Loads every persona from tester_profiles/ and every company from companies/,
sends the persona + company data to Gemma4 via the master prompt, and stores
the generated system prompts in simulation_results.xlsx.

Usage (from the personas/ directory):
    python run_simulation.py

Or from the project root:
    python personas/run_simulation.py
"""

import re
import sys
import time
from pathlib import Path

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
import requests

# ---------------------------------------------------------------------------
# Path setup — works whether run from personas/ or the project root
# ---------------------------------------------------------------------------

PERSONAS_DIR = Path(__file__).parent
sys.path.insert(0, str(PERSONAS_DIR))

from master_prompt import build_generation_request  # noqa: E402

TESTER_PROFILES_DIR = PERSONAS_DIR / "tester_profiles"
COMPANIES_DIR = PERSONAS_DIR / "companies"
COMPANIES_OVERVIEW = PERSONAS_DIR / "companies_overview.md"
RESULTS_FILE = PERSONAS_DIR / "simulation_results.xlsx"

# ---------------------------------------------------------------------------
# Ollama client
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gemma4:latest"


def query_gemma(system_msg: str, user_msg: str) -> str:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "stream": False,
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=180)
    response.raise_for_status()
    return response.json()["message"]["content"]


# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------


def _extract_section(md: str, *header_variants: str) -> str:
    """
    Return the body of the first ## section whose heading matches any of the
    provided variants (case-insensitive, partial match allowed).
    Stops at the next ## heading.
    """
    lines = md.splitlines()
    body: list[str] = []
    capturing = False

    for line in lines:
        if line.startswith("## "):
            heading = line[3:].strip().lower()
            if capturing:
                break
            if any(v.lower() in heading for v in header_variants):
                capturing = True
            continue
        if capturing:
            body.append(line)

    return "\n".join(body).strip()


def _first_sentence(text: str) -> str:
    """Return the first sentence of a block of text."""
    text = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
    match = re.match(r"([^.!?]+[.!?])", text)
    return match.group(1).strip() if match else text[:120]


def _extract_inline(md: str, bold_label: str) -> str:
    """Extract the value after a **Label:** pattern anywhere in the file."""
    pattern = rf"\*\*{re.escape(bold_label)}:\*\*\s*(.+)"
    match = re.search(pattern, md)
    return match.group(1).strip() if match else ""


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def parse_persona(path: Path) -> dict:
    md = path.read_text(encoding="utf-8")

    name_match = re.match(r"#\s+Persona:\s+(.+)", md)
    name = name_match.group(1).strip() if name_match else path.stem

    identity_block = _extract_section(md, "identity")
    behavioural = _extract_section(md, "behavioural profile", "behavioral profile")
    summary = _extract_section(md, "summary")
    tone_raw = _extract_section(md, "tone")

    brief_identity = (
        _first_sentence(summary) if summary else _first_sentence(identity_block)
    )
    brief_identity = re.sub(r"\*\*", "", brief_identity)

    full_identity = "\n\n".join(filter(None, [identity_block, behavioural]))

    return {
        "name": name,
        "brief_identity": brief_identity,
        "identity_block": full_identity or identity_block,
        "tone": re.sub(r"\*\*", "", tone_raw).strip(),
        "interaction_rules": _extract_section(md, "interaction rules"),
        "failure_patterns": _extract_section(md, "failure patterns"),
        "role_anchor": _extract_section(md, "role anchor"),
    }


def _parse_industry_map() -> dict[str, str]:
    """Read companies_overview.md and build {company_name: industry} mapping."""
    industry_map: dict[str, str] = {}
    if not COMPANIES_OVERVIEW.exists():
        return industry_map
    md = COMPANIES_OVERVIEW.read_text(encoding="utf-8")
    for block in re.split(r"^---", md, flags=re.MULTILINE):
        name_match = re.search(r"##\s+C\d+\s+[—–-]+\s+(.+)", block)
        industry_match = re.search(r"\*\*Industry:\*\*\s*(.+)", block)
        if name_match and industry_match:
            industry_map[name_match.group(1).strip()] = industry_match.group(1).strip()
    return industry_map


_INDUSTRY_MAP: dict[str, str] = {}


def parse_company(path: Path) -> dict:
    global _INDUSTRY_MAP
    if not _INDUSTRY_MAP:
        _INDUSTRY_MAP = _parse_industry_map()

    md = path.read_text(encoding="utf-8")

    name_match = re.match(r"#\s+Company Profile:\s+(.+)", md)
    company_name = name_match.group(1).strip() if name_match else path.stem

    agent_section = _extract_section(md, "2.", "the agent")
    bot_name = _extract_inline(agent_section, "Name")
    if not bot_name:
        bot_name = f"{company_name} Assistant"

    industry = _INDUSTRY_MAP.get(company_name, "")
    capabilities = _extract_section(md, "3.", "core capabilities")
    test_scenarios = _extract_section(md, "4.", "test case parameters")

    return {
        "bot_name": bot_name,
        "company_name": company_name,
        "industry": industry,
        "capabilities_summary": capabilities,
        "test_scenarios": test_scenarios,
    }


# ---------------------------------------------------------------------------
# Excel export
# ---------------------------------------------------------------------------

HEADERS = ["#", "Persona", "Company", "Elapsed (s)", "Generated System Prompt", "Error"]
HEADER_FILL = PatternFill("solid", fgColor="1F3864")
HEADER_FONT = Font(bold=True, color="FFFFFF")
ERROR_FILL = PatternFill("solid", fgColor="FFE0E0")


def _init_excel() -> None:
    """Create the results file with headers and column widths."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Generated Prompts"

    for col, header in enumerate(HEADERS, start=1):
        cell = ws.cell(row=1, column=col, value=header)
        cell.font = HEADER_FONT
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(horizontal="center", vertical="center")

    ws.row_dimensions[1].height = 20
    ws.column_dimensions["A"].width = 5  # #
    ws.column_dimensions["B"].width = 16  # Persona
    ws.column_dimensions["C"].width = 18  # Company
    ws.column_dimensions["D"].width = 12  # Elapsed
    ws.column_dimensions["E"].width = 120  # Generated System Prompt
    ws.column_dimensions["F"].width = 40  # Error

    wb.save(RESULTS_FILE)


def _append_row(result: dict, row_num: int) -> None:
    """Append a single result row to the existing Excel file."""
    wb = openpyxl.load_workbook(RESULTS_FILE)
    ws = wb.active
    is_error = bool(result.get("error"))
    row = [
        row_num,
        result.get("persona", ""),
        result.get("company", ""),
        result.get("elapsed_s", ""),
        result.get("generated_prompt", "") if not is_error else "",
        result.get("error", ""),
    ]
    excel_row = row_num + 1  # +1 for header row
    for col, value in enumerate(row, start=1):
        cell = ws.cell(row=excel_row, column=col, value=value)
        cell.alignment = Alignment(wrap_text=True, vertical="top")
        if is_error:
            cell.fill = ERROR_FILL
    wb.save(RESULTS_FILE)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_all() -> None:
    personas = sorted(TESTER_PROFILES_DIR.glob("p*.md"))
    companies = sorted(COMPANIES_DIR.glob("c*.md"))

    if not personas:
        print(f"No persona files found in {TESTER_PROFILES_DIR}")
        sys.exit(1)
    if not companies:
        print(f"No company files found in {COMPANIES_DIR}")
        sys.exit(1)

    persona_dicts = [parse_persona(p) for p in personas]
    company_dicts = [parse_company(c) for c in companies]

    total = len(persona_dicts) * len(company_dicts)
    print(
        f"Generating prompts: {len(persona_dicts)} personas × {len(company_dicts)} companies = {total} combinations\n"
    )

    _init_excel()
    count = 0

    for persona in persona_dicts:
        for company in company_dicts:
            count += 1
            label = (
                f"[{count:02d}/{total}] {persona['name']} × {company['company_name']}"
            )
            print(f"{label} ... ", end="", flush=True)

            try:
                system_msg, user_msg = build_generation_request(persona, company)
                t0 = time.time()
                generated_prompt = query_gemma(system_msg, user_msg)
                elapsed = round(time.time() - t0, 1)

                print(f"done ({elapsed}s)")
                result = {
                    "persona": persona["name"],
                    "company": company["company_name"],
                    "elapsed_s": elapsed,
                    "generated_prompt": generated_prompt,
                }

            except requests.exceptions.ConnectionError:
                print("FAILED — Ollama not reachable at http://localhost:11434")
                print("  Make sure Ollama is running: ollama serve")
                sys.exit(1)
            except Exception as exc:
                print(f"ERROR — {exc}")
                result = {
                    "persona": persona["name"],
                    "company": company["company_name"],
                    "error": str(exc),
                }

            _append_row(result, count)

    print(f"\n{'=' * 60}")
    print(f"Done. {count} prompts generated and saved to {RESULTS_FILE.name}")


if __name__ == "__main__":
    run_all()
