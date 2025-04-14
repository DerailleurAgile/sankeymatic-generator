import pandas as pd
import re
import argparse
import datetime
from typing import Dict, List, Tuple

# === Command-Line Argument Parsing ===
def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate a SankeyMatic diagram from Leadership Survey data.")
    parser.add_argument("--topraw", action="store_true", help="Show only the top 5 individual correlations (raw cells) between practices and symptoms.")
    parser.add_argument("--toppractices", action="store_true", help="Filter to show only the top contributing practices to the symptoms.")
    parser.add_argument("--allpractices", action="store_true", help="Include all practices (unfiltered) in the aggregated map (in addition to filtering modes).")
    return parser.parse_args()

# === Color Management ===
def lighten_color(hex_color: str, percentage: float = 0.2) -> str:
    """Lighten a hex color by a given percentage."""
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    rgb_light = tuple(min(255, int(c + (255 - c) * percentage)) for c in rgb)
    return f"#{rgb_light[0]:02x}{rgb_light[1]:02x}{rgb_light[2]:02x}"

def define_practice_colors() -> Tuple[Dict[str, str], Dict[str, str]]:
    """Define colors for Practice Categories and Practices."""
    practice_category_colors = {
        "Reactive Behaviors": "#1f77b4",
        "Fear-Inducing Practices": "#2ca02c",
        "Siloed Structures": "#9467bd",
        "Target-Driven Approaches": "#ff7f0e",
        "Cost-Focused Practices": "#d62728"
    }
    practice_colors = {
        "Q18": lighten_color(practice_category_colors["Reactive Behaviors"], 0.3),
        "Q25": lighten_color(practice_category_colors["Reactive Behaviors"], 0.5),
        "Q19": lighten_color(practice_category_colors["Fear-Inducing Practices"], 0.3),
        "Q20": lighten_color(practice_category_colors["Fear-Inducing Practices"], 0.5),
        "Q21": lighten_color(practice_category_colors["Siloed Structures"], 0.3),
        "Q22": lighten_color(practice_category_colors["Siloed Structures"], 0.5),
        "Q23": lighten_color(practice_category_colors["Target-Driven Approaches"], 0.3),
        "Q24": lighten_color(practice_category_colors["Target-Driven Approaches"], 0.5),
        "Q26": lighten_color(practice_category_colors["Cost-Focused Practices"], 0.3),
        "Q27": lighten_color(practice_category_colors["Cost-Focused Practices"], 0.5)
    }
    return practice_category_colors, practice_colors

def define_symptom_colors() -> Tuple[Dict[str, str], Dict[str, str]]:
    """Define colors for Symptom Categories and Symptoms."""
    symptom_category_colors = {
        "Cultural Symptoms": "#9e9ac8",
        "Leadership Symptoms": "#bdbdbd",
        "Operational Symptoms": "#d9d574",
        "Outcome Symptoms": "#74a9cf"
    }
    symptom_colors = {
        "Q1": "#bcbddc",
        "Q2": "#bcbddc",
        "Q3": "#9e9ac8",
        "Q4": "#bcbddc",
        "Q5": "#bcbddc",
        "Q6": "#bcbddc",
        "Q7": "#bdbdbd",
        "Q8": "#bdbdbd",
        "Q9": "#bdbdbd",
        "Q10": "#bdbdbd",
        "Q11": "#d9d574",
        "Q12": "#d9d574",
        "Q13": "#d9d574",
        "Q14": "#d9d574",
        "Q15": "#d9d574",
        "Q16": "#74a9cf",
        "Q17": "#74a9cf"
    }
    return symptom_category_colors, symptom_colors

# === Data Loading and Cleaning ===
def load_excel_file(file_path: str) -> pd.DataFrame:
    """Load the Excel file and return the DataFrame."""
    print("Loading Excel file...")
    try:
        df = pd.read_excel(file_path, sheet_name=0)
        print("Excel file loaded. Initial shape:", df.shape)
        return df
    except Exception as e:
        print(f"❌ Error loading Excel file: {e}")
        raise

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the DataFrame by stripping strings from text entries."""
    print("Cleaning data...")
    df = df.apply(lambda col: col.map(lambda x: str(x).strip() if isinstance(x, str) else x))
    df.columns = df.columns.map(str)
    print("Data cleaned. Shape after cleaning:", df.shape)
    return df

def extract_qcode_and_label(text: str) -> Tuple[str, str]:
    """Extract Q-code and label from a text string."""
    text = str(text).strip().replace("–", "-").replace("—", "-")
    match = re.match(r'(Q\d{1,2})\s*[:\-–]?\s*(.*)', text)
    if match:
        qcode = match.group(1)
        label = f"{qcode}: {match.group(2).strip()}"
        return qcode, label
    return None, None

# --- Aggregated Modes (Used by all aggregated modes) ---
def process_rows(df: pd.DataFrame, first_col: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Process row labels (assumed in first column) and set index."""
    row_label_map = {}
    qcodes = []
    for raw in df[first_col]:
        qcode, full_label = extract_qcode_and_label(raw)
        if qcode:
            row_label_map[qcode] = full_label
            qcodes.append(qcode)
        else:
            qcodes.append(None)
    df["QCODE"] = qcodes
    print("QCODE column added. Shape:", df.shape)
    df = df[~df["QCODE"].isnull()].copy()
    print("Rows with null QCODE removed. Shape:", df.shape)
    if df.empty:
        raise ValueError("No valid Q-codes found in rows.")
    df.set_index("QCODE", inplace=True)
    print("Set QCODE as index. Shape:", df.shape)
    return df, row_label_map

def process_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Process column labels (assumed to contain Q-codes) and rename columns."""
    col_label_map = {}
    valid_cols = []
    for col in df.columns:
        qcode, full_label = extract_qcode_and_label(str(col))
        if qcode:
            col_label_map[qcode] = full_label
            valid_cols.append((qcode, col))
    df = df[[orig for (_, orig) in valid_cols]]
    print("Filtered valid columns. Shape:", df.shape)
    if df.empty:
        raise ValueError("No valid columns found.")
    df.columns = [qcode for (qcode, _) in valid_cols]
    print("Renamed columns. Shape:", df.shape)
    return df, col_label_map

def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and filter DataFrame for expected Q-codes."""
    valid_symptoms = [f"Q{i}" for i in range(1, 18)]
    valid_practices = [f"Q{i}" for i in range(18, 28)]
    print("Valid symptoms:", valid_symptoms)
    print("Valid practices:", valid_practices)
    df = df.loc[df.index.intersection(valid_symptoms), df.columns.intersection(valid_practices)]
    print("Filtered for valid symptoms and practices. Shape:", df.shape)
    if df.empty:
        raise ValueError("DataFrame is empty after filtering.")
    print("\n--- FINAL MATRIX SHAPE ---")
    print(df.shape)
    print("Q22 in columns?", "Q22" in df.columns)
    print("Q1 in index?", "Q1" in df.index)
    return df

def define_groups() -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Define practice and symptom groups (for aggregated modes)."""
    practice_groups = {
        "Reactive Behaviors": ["Q18", "Q25"],
        "Fear-Inducing Practices": ["Q19", "Q20"],
        "Siloed Structures": ["Q21", "Q22"],
        "Target-Driven Approaches": ["Q23", "Q24"],
        "Cost-Focused Practices": ["Q26", "Q27"]
    }
    symptom_groups = {
        "Cultural Symptoms": ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"],
        "Leadership Symptoms": ["Q7", "Q8", "Q9", "Q10"],
        "Operational Symptoms": ["Q11", "Q12", "Q13", "Q14", "Q15"],
        "Outcome Symptoms": ["Q16", "Q17"]
    }
    return practice_groups, symptom_groups

def calculate_practice_to_symptom_flows(df: pd.DataFrame,
                                        symptom_groups: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
    """Calculate aggregated flows from practices to symptom categories."""
    practice_to_symptom_group = {practice: {group: 0 for group in symptom_groups} for practice in df.columns}
    for practice in df.columns:
        for symptom in df.index:
            try:
                num = float(str(df.at[symptom, practice]).strip())
                if num > 0:
                    symptom_group = next(group for group, symptoms in symptom_groups.items() if symptom in symptoms)
                    practice_to_symptom_group[practice][symptom_group] += num
            except Exception as e:
                print(f"Warning: Skipping {symptom} to {practice}: {df.at[symptom, practice]} (Error: {e})")
    return practice_to_symptom_group

def calculate_top_correlations(practice_to_symptom_group: Dict[str, Dict[str, float]],
                               symptom_groups: Dict[str, List[str]],
                               col_label_map: Dict[str, str],
                               top_n: int = 5) -> List[Tuple[str, str, float]]:
    """Calculate top aggregated correlations (for aggregated modes)."""
    correlations = []
    for symptom_group in symptom_groups:
        for practice in practice_to_symptom_group:
            total = practice_to_symptom_group[practice].get(symptom_group, 0)
            if total > 0:
                correlations.append((practice, symptom_group, total))
    sorted_correlations = sorted(correlations, key=lambda x: x[2], reverse=True)[:top_n]
    print(f"\nTop {top_n} aggregated correlations:")
    for practice, symptom_group, value in sorted_correlations:
        print(f"{col_label_map.get(practice, practice)} -> {symptom_group}: {value:.2f}")
    return sorted_correlations

def calculate_top_raw_correlations(df: pd.DataFrame,
                                   top_n: int = 5) -> List[Tuple[str, str, float]]:
    """Calculate top raw cell correlations (for --topraw mode)."""
    correlations = []
    for practice in df.columns:
        for symptom in df.index:
            try:
                num = float(str(df.at[symptom, practice]).strip())
                if num > 0:
                    correlations.append((practice, symptom, num))
            except Exception as e:
                print(f"Warning: Skipping {symptom} to {practice}: {df.at[symptom, practice]} (Error: {e})")
    sorted_correlations = sorted(correlations, key=lambda x: x[2], reverse=True)[:top_n]
    print(f"\nTop {top_n} raw correlations:")
    for practice, symptom, value in sorted_correlations:
        print(f"{practice} -> {symptom}: {value:.2f}")
    return sorted_correlations

# --- New Raw Flow Generators (Direct Source→Target) ---
def generate_raw_sankey_lines(raw_df: pd.DataFrame,
                              col_label_map: Dict[str, str],
                              row_label_map: Dict[int, str]) -> Tuple[List[str], List[str], List[str]]:
    """
    Generate raw Sankey flow lines directly from raw_df.
    Each nonzero cell becomes a flow: <Source> [value] <Target>.
    Returns (empty_layer, flows, empty_layer).
    """
    raw_flows = []
    for i, row in raw_df.iterrows():
        for col in raw_df.columns:
            try:
                value = float(str(row[col]).strip())
                if value > 0:
                    raw_flows.append(f"{col_label_map.get(col, str(col))} [{value:.2f}] {row_label_map.get(i, str(i))}")
            except Exception:
                continue
    return ([], raw_flows, [])

def generate_color_lines_raw(practice_colors: Dict[str, str],
                             symptom_colors: Dict[str, str],
                             col_label_map: Dict[str, str],
                             row_label_map: Dict[int, str],
                             raw_df: pd.DataFrame) -> List[str]:
    """
    Generate color directives for raw mode.
    Only outputs colors for practices and symptoms that have nonzero flow.
    """
    color_lines = ["//==============================", "// NODE COLOR DEFINITIONS", "//==============================", ""]
    for col in raw_df.columns:
        total = pd.to_numeric(raw_df[col], errors='coerce').fillna(0).sum()
        if total > 0:
            color = practice_colors.get(col, "#000000")
            display_label = col_label_map.get(col, str(col))
            color_lines.append(f":{display_label} {color}")
    for i in raw_df.index:
        total = pd.to_numeric(raw_df.loc[i], errors='coerce').fillna(0).sum()
        if total > 0:
            full_label = row_label_map.get(i, str(i))
            qcode, label = extract_qcode_and_label(full_label)
            if qcode is None:
                key = full_label
                display_label = full_label
            else:
                key = qcode
                display_label = label
            color = symptom_colors.get(key, "#000000")
            color_lines.append(f":{display_label} {color}")
    return color_lines

# --- Aggregated Color Lines Generator (Rolled Back) ---
def generate_color_lines_aggregated(
    practice_groups: Dict[str, List[str]],
    practice_category_colors: Dict[str, str],
    practice_colors: Dict[str, str],
    symptom_groups: Dict[str, List[str]],
    symptom_category_colors: Dict[str, str],
    symptom_colors: Dict[str, str],
    col_label_map: Dict[str, str],
    row_label_map: Dict[str, str],
    df: pd.DataFrame,
    top_practices_set: set = None
) -> List[str]:
    """
    Generate color definitions for aggregated mode.
    Only output a color directive for a node (category, practice, or symptom)
    if its total flow is greater than zero.
    """
    color_lines = [
        "//==============================",
        "// NODE COLOR DEFINITIONS",
        "//==============================",
        ""
    ]
    # 1. Practice Categories
    for category, practices in practice_groups.items():
        relevant_practices = [p for p in practices if p in df.columns]
        if top_practices_set is not None:
            relevant_practices = [p for p in relevant_practices if p in top_practices_set]
        total_cat = 0
        for p in relevant_practices:
            try:
                total_cat += df[p].astype(float).sum()
            except:
                pass
        if total_cat > 0:
            color_lines.append(f":{category} {practice_category_colors.get(category, '#000000')}")
    
    # 2. Individual Practices
    for category, practices in practice_groups.items():
        for p in practices:
            if p not in df.columns:
                continue
            if top_practices_set and p not in top_practices_set:
                continue
            total_flow = df[p].astype(float).sum()
            if total_flow > 0:
                disp_label = col_label_map.get(p, p)
                color = practice_colors.get(p, "#000000")
                color_lines.append(f":{disp_label} {color}")
    
    # 3. Symptom Categories
    for cat, color in symptom_category_colors.items():
        qcodes = symptom_groups.get(cat, [])
        qcodes = [q for q in qcodes if q in df.index]
        total = 0
        for q in qcodes:
            try:
                total += df.loc[q].astype(float).sum()
            except:
                pass
        if total > 0:
            color_lines.append(f":{cat} {color}")
    
    # 4. Individual Symptoms
    for cat, qcodes in symptom_groups.items():
        for q in qcodes:
            if q not in df.index:
                continue
            try:
                total = df.loc[q].astype(float).sum()
            except:
                total = 0
            if total > 0:
                disp_label = row_label_map.get(q, q)
                color = symptom_colors.get(q, "#000000")
                color_lines.append(f":{disp_label} {color}")
    
    return color_lines

# --- Aggregated Sankey Lines Generator ---
def generate_sankey_lines_aggregated(df: pd.DataFrame,
                                     practice_groups: Dict[str, List[str]],
                                     symptom_groups: Dict[str, List[str]],
                                     practice_to_symptom_group: Dict[str, Dict[str, float]],
                                     col_label_map: Dict[str, str],
                                     row_label_map: Dict[str, str],
                                     top_raw: bool = False,
                                     sorted_raw_correlations: List[Tuple[str, str, float]] = None,
                                     sorted_correlations: List[Tuple[str, str, float]] = None,
                                     top_practices_set: set = None) -> Tuple[List[str], List[str], List[str]]:
    """
    Generate aggregated Sankey flow lines.
    """
    if top_raw and sorted_raw_correlations:
        flows = []
        for practice, symptom, value in sorted_raw_correlations:
            flows.append(f"{col_label_map.get(practice, practice)} [{value:.2f}] {row_label_map.get(symptom, symptom)}")
        return ([], flows, [])
    else:
        practice_category_lines = ["\n//==============================", "// PRACTICE CATEGORY → PRACTICES", "//==============================", ""]
        for group, practices in practice_groups.items():
            category_total = 0
            for practice in practices:
                if top_practices_set and practice not in top_practices_set:
                    continue
                total = sum(practice_to_symptom_group[practice][sg] for sg in symptom_groups)
                category_total += total
            if category_total == 0:
                continue
            for practice in practices:
                if top_practices_set and practice not in top_practices_set:
                    continue
                total = sum(practice_to_symptom_group[practice][sg] for sg in symptom_groups)
                if total > 0:
                    practice_category_lines.append(f"{group} [{total:.2f}] {col_label_map.get(practice, practice)}")
        practice_to_symptom_lines = ["\n//==============================", "// PRACTICES → SYMPTOM CATEGORIES", "//==============================", ""]
        for sg in symptom_groups:
            for practice in df.columns:
                if top_practices_set and practice not in top_practices_set:
                    continue
                total = practice_to_symptom_group[practice].get(sg, 0)
                if total > 0:
                    practice_to_symptom_lines.append(f"{col_label_map.get(practice, practice)} [{total:.2f}] {sg}")
        symptom_category_lines = ["\n//==============================", "// SYMPTOM CATEGORIES → INDIVIDUAL SYMPTOMS", "//==============================", ""]
        for symptom in df.index:
            symptom_group = next(group for group, symptoms in symptom_groups.items() if symptom in symptoms)
            total = 0
            relevant_practices = [practice for practice in df.columns if (not top_practices_set or practice in top_practices_set)]
            for practice in relevant_practices:
                try:
                    num = float(str(df.at[symptom, practice]).strip())
                    total += num
                except:
                    continue
            if total > 0:
                symptom_category_lines.append(f"{symptom_group} [{total:.2f}] {row_label_map.get(symptom, symptom)}")
        return practice_category_lines, practice_to_symptom_lines, symptom_category_lines

def write_sankey_output(sankey_lines: List[str],
                        color_lines: List[str],
                        practice_category_lines: List[str],
                        practice_to_symptom_lines: List[str],
                        symptom_category_lines: List[str],
                        output_file: str = "sankeymatic_output.txt",
                        header_lines: List[str] = None) -> None:
    """Write the SankeyMatic output to a file."""
    if header_lines is None:
        header_lines = []
    # all_lines = header_lines + ["// === SankeyMATIC Diagram Inputs ==="] + color_lines + practice_category_lines + practice_to_symptom_lines + symptom_category_lines
    all_lines = header_lines + color_lines + practice_category_lines + practice_to_symptom_lines + symptom_category_lines
    
    if all_lines:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(all_lines))
        print(f"\n✅ Wrote {len(color_lines)-2} color directives and {len(practice_category_lines + practice_to_symptom_lines + symptom_category_lines)-6} Sankey lines to {output_file}")
    else:
        print("\n⚠️ No valid Sankey entries generated.")

def main():
    args = parse_arguments()
    # Determine mode. Default mode is "toppractices".
    if args.topraw:
        mode = "topraw"
    elif args.toppractices:
        mode = "toppractices"
    else:
        mode = "toppractices"  # default filtered mode
    # If --allpractices is provided, we do not filter by top practices.
    show_all_practices = args.allpractices

    # We'll use aggregated mode for all these options.
    aggregated_mode = True

    # Step 1: Define colors.
    practice_category_colors, practice_colors = define_practice_colors()
    symptom_category_colors, symptom_colors = define_symptom_colors()

    # Step 2: Load and clean data.
    df = load_excel_file("Leadership Survey Matrices.xlsx")
    df = clean_dataframe(df)
    print("\n--- RAW FIRST COLUMN VALUES (first 20) ---")
    for i, val in enumerate(df.iloc[:20, 0]):
        print(f"{i+1:02d}: {repr(val)}")

    # In aggregated mode, process rows/columns and validate.
    if aggregated_mode:
        first_col = df.columns[0]
        df, row_label_map = process_rows(df, first_col)
        df, col_label_map = process_columns(df)
        df = validate_dataframe(df)
    else:
        row_label_map = {}
        col_label_map = {}

    # For aggregated modes, define groups and calculate flows.
    practice_groups, symptom_groups = define_groups()
    practice_to_symptom_group = calculate_practice_to_symptom_flows(df, symptom_groups)

    # Step 3: Determine filtering mode and calculate correlations if needed.
    top_practices_set = None
    sorted_raw_correlations = None
    sorted_correlations = None

    if mode == "topraw":
        sorted_raw_correlations = calculate_top_raw_correlations(df, top_n=5)
    elif mode == "toppractices":
        # If we're not showing all practices, we filter top practices.
        if not show_all_practices:
            top_practices_set = get_top_practices(df, top_n=5)
        sorted_correlations = calculate_top_correlations(practice_to_symptom_group, symptom_groups, col_label_map)
    elif mode == "allpractices":
        top_practices_set = None
        sorted_correlations = calculate_top_correlations(practice_to_symptom_group, symptom_groups, col_label_map)

    # Step 4: Generate Sankey lines (aggregated).
    if mode == "topraw":
        practice_category_lines, practice_to_symptom_lines, symptom_category_lines = generate_sankey_lines_aggregated(
            df, practice_groups, symptom_groups, practice_to_symptom_group,
            col_label_map, row_label_map,
            top_raw=True, sorted_raw_correlations=sorted_raw_correlations,
            sorted_correlations=None, top_practices_set=None
        )
    else:
        practice_category_lines, practice_to_symptom_lines, symptom_category_lines = generate_sankey_lines_aggregated(
            df, practice_groups, symptom_groups, practice_to_symptom_group,
            col_label_map, row_label_map,
            top_raw=False, sorted_raw_correlations=None, sorted_correlations=sorted_correlations,
            top_practices_set=(top_practices_set if not show_all_practices else None)
        )

    # Step 5: Generate color lines using the aggregated color generator.
    color_lines = generate_color_lines_aggregated(
        practice_groups, practice_category_colors, practice_colors,
        symptom_groups, symptom_category_colors, symptom_colors,
        col_label_map, row_label_map, df,
        top_practices_set=(top_practices_set if not show_all_practices else None)
    )

    # Step 6: Create header with mode description and current time.
    if mode == "topraw":
        mode_desc = "TOP-RAW --> SYMPTOMS"
    elif mode == "toppractices":
        if show_all_practices:
            mode_desc = "ALL-PRACTICES --> SYMPTOMS"
        else:
            mode_desc = "TOP-PRACTICES --> SYMPTOMS"
    else:
        mode_desc = "TOP-PRACTICES --> SYMPTOMS"
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header_lines = [f"// {mode_desc}", f"// Rendered at: {current_time}"]

    # Step 7: Write output.
    write_sankey_output([], color_lines, practice_category_lines, practice_to_symptom_lines, symptom_category_lines,
                        header_lines=header_lines)

# Utility function for top practices
def get_top_practices(df: pd.DataFrame, top_n: int = 5) -> set:
    totals = {}
    for col in df.columns:
        try:
            totals[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).sum()
        except Exception as e:
            totals[col] = 0
    sorted_practices = sorted(totals.items(), key=lambda x: x[1], reverse=True)[:top_n]
    print(f"\nTop {top_n} practices by total contribution:")
    for pr, total in sorted_practices:
        print(f"{pr}: {total:.2f}")
    return {pr for pr, total in sorted_practices}

if __name__ == "__main__":
    main()
