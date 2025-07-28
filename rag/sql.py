import json
import sqlite3
from typing import Optional, List, Dict, Any
# from rag.utils import best_string_match
from rapidfuzz import process, fuzz


def best_string_match(query: str, candidates: list, threshold: float = 90.0, index: int = 1) -> Optional[tuple]:
    """
    Compare `query` to each string in `candidates`.
    Returns (best_match, best_score) if score >= threshold, else None.
    """
    if not query or not candidates:
        return None
    # Normalize candidates (optional, but can improve matching for your use case)
    candidate_strings = [c[index].strip().lower() for c in candidates]
    print(f"candidate_strings={candidate_strings}")
    query = query.strip().lower()

    # Get the best match and its score
    result = process.extractOne(query, candidate_strings, scorer=fuzz.WRatio)
    if result and result[1] >= threshold:
        # Return the whole tuple from the list
        print(f"Best match found: {result[0]} with score {result[1]}")
        return [c for c in candidates if c[index].strip().lower() == result[0]][0]
    else:
        return None
    
class DataManager:
    def __init__(self, db_file: str, initial_json: Optional[str] = None):
        self.db_file = db_file
        self.conn = sqlite3.connect(db_file)
        self.cursor = self.conn.cursor()
        self.create_table()
        if initial_json and self.is_table_empty():
            print(f"Table empty: initializing with {initial_json}")
            result = self.insert_mapping_bulk_json(initial_json)
            # print(result)
    def is_table_empty(self) -> bool:
            self.cursor.execute("SELECT COUNT(*) FROM concept_mappings")
            count = self.cursor.fetchone()[0]
            # print(f"Table 'concept_mappings' is empty: {count == 0}")
            return count == 0
    def create_table(self):
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS concept_mappings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            variable_name TEXT UNIQUE,
            standard_label TEXT,
            concept_code TEXT,
            omop_id INTEGER
        );
        """
        self.cursor.execute(create_table_sql)
        self.conn.commit()

    def insert_mapping(
        self,
        variable_name: Optional[str],

        standard_label: Optional[str],
        concept_code: Optional[str],
        omop_id: Optional[int],
    ) -> Dict[str, Any]:
        try:
            self.cursor.execute(
                """
                INSERT INTO concept_mappings (
                    variable_name,  standard_label, concept_code, omop_id
                ) VALUES (?, ?, ?, ?)
                """,
                (variable_name, standard_label, concept_code, omop_id),
            )
            self.conn.commit()
            return {"status": "success", "message": "Row inserted."}
        except sqlite3.IntegrityError:
            return {"status": "info", "message": "Duplicate row skipped."}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def insert_many(self, mappings: List[Dict[str, Any]]) -> None:
        for mapping in mappings:
            print(self.insert_mapping(**mapping))

    def insert_mapping_bulk_json(self, json_file_path: str) -> Dict[str, Any]:
        try:
            with open(json_file_path, "r") as f:
                data = json.load(f)
            # Expecting: {"database_data": [ {...}, {...}, ... ]}
            if "database_data" not in data or not isinstance(data["database_data"], list):
                return {"status": "error", "message": "JSON must contain a 'database_data' list."}
            raw_mappings = data["database_data"]
            # Deduplicate: create a set of tuples for unique fields
            seen = set()
            unique_mappings = []
            for entry in raw_mappings:
                # Only consider the relevant fields for deduplication
                dedup_tuple = (
                    entry.get("variable_label"),
                    entry.get("standard_label"),
                    entry.get("concept_code"),
                    entry.get("omop_id")
                )

                if dedup_tuple not in seen:
                    seen.add(dedup_tuple)
                    unique_mappings.append({
                        "variable_name": entry.get("variable_label").strip().lower(),
                        "standard_label": entry.get("standard_label").strip().lower(),
                        "concept_code": entry.get("code"),
                        "omop_id": entry.get("omop_id"),
                    })
            self.insert_many(unique_mappings)
            return {"status": "success", "message": f"Bulk insert completed. {len(unique_mappings)} unique rows inserted."}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def find_by_variable(self, variable_name: str) -> Optional[tuple]:
        # 1. Try exact case-insensitive match first
        self.cursor.execute(
            "SELECT * FROM concept_mappings WHERE LOWER(variable_name) = ?",
            (variable_name.lower(),)
        )
        result = self.cursor.fetchone()
        if result:
            return result

        # 2. If not found, fall back to substring (optional)
        self.cursor.execute(
            "SELECT * FROM concept_mappings WHERE LOWER(variable_name) LIKE ?",
            (f"%{variable_name.lower()}%",)
        )
        results = self.cursor.fetchall()
        print(f"Results found by variable name: {results}")
        result = best_string_match(variable_name, results, threshold=92)  # Use index=1 for 'standard_label'
        print(f"Found best match entity {result} by variable Name: {variable_name}")
        # Return all matches for manual review or further filtering
        return result if results else None

    def find_by_label(self, label: str) -> Optional[tuple]:
        # 1. Try exact case-insensitive match first
        self.cursor.execute(
            "SELECT * FROM concept_mappings WHERE LOWER(standard_label) = ?",
            (label.lower(),)
        )
        result = self.cursor.fetchone()
        if result:
            return result

        # 2. If not found, fall back to substring (optional)
        self.cursor.execute(
            "SELECT * FROM concept_mappings WHERE LOWER(standard_label) LIKE ?",
            (f"%{label.lower()}%",)
        )
        results = self.cursor.fetchall()
        print(f"Results found by label: {results}")
        result = best_string_match(label, results, threshold=92, index=2)  # Use index=1 for 'standard_label'
        print(f"Found best match entity {result} by variable label: {label}")
        return result

    def close_connection(self):
        if self.conn:
            self.conn.close()
            print("Database connection closed.")
        else:
            print("No active database connection to close.")
            
    def delete_mapping(self, variable_name: str) -> Dict[str, Any]:
        try:
            self.cursor.execute(
                "DELETE FROM concept_mappings WHERE LOWER(variable_name) = ?",
                (variable_name.lower(),)
            )
            self.conn.commit()
            if self.cursor.rowcount > 0:
                return {"status": "success", "message": "Row deleted."}
            else:
                return {"status": "info", "message": "No matching row found to delete."}
        except Exception as e:
            return {"status": "error", "message": str(e)}


# Example usage
if __name__ == "__main__":
    db = DataManager("variables.db", initial_json="/Users/komalgilani/Desktop/cde_mapper/data/input/mapping_templates.json")

    # Insert bulk from JSON file
    # result = db.insert_mapping_bulk_json("/Users/komalgilani/Desktop/cde_mapper/data/input/mapping_templates.json")
    # print(result)

    # Example query
    
    

    print(db.find_by_variable("36 months"))

    print(db.find_by_label("24 months"))
    
    
    db.delete_mapping("24 months")
    db.delete_mapping("6 months")
    db.delete_mapping("3 months")
    db.delete_mapping("18 months")
    db.delete_mapping("mid-regional proadrenomedullin (mr-proadm)")
    
    # insert_result = db.insert_mapping(
    #     variable_name="12 months",
    #     standard_label="Follow-up 1 year",
    #     concept_code="snomed:183627004",
    #     omop_id=4081746
    # )
    
    insert_result = db.insert_mapping(
        variable_name="3 months",
        standard_label="Follow-up 3 months",
        concept_code="snomed:200521000000107",
        omop_id=44789369
    )
    print(insert_result)
    insert_result = db.insert_mapping(
        variable_name="6 months",
        standard_label="Follow-up 6 months",
        concept_code="snomed:300042001",
        omop_id=4103967
    )
    print(insert_result)
    # insert_result = db.insert_mapping(
    #     variable_name="18 months",
    #     standard_label="Follow-up 18 months",
    #     concept_code="snomed:44789050",
    #     omop_id=44789050
    # )
    print(db.insert_mapping(
        variable_name="24 months",
        standard_label="Follow-up 2 years",
        concept_code="snomed:199581000000104",
        omop_id=44789049
    ))
    
    insert_result = db.insert_mapping(
        variable_name="dose at night",
        standard_label="Night time",
        concept_code="snomed:2546009",
        omop_id=4102546
    )
    insert_result = db.insert_mapping(
        variable_name="dose in the morning",
        standard_label="morning",
        concept_code="snomed:73775008",
        omop_id=4252249
    )
    insert_result = db.insert_mapping(
        variable_name="dose at noon",
        standard_label="noon",
        concept_code="snomed:71997007",
        omop_id=4215920
    )
    insert_result = db.insert_mapping(
        variable_name="dose at evening",
        standard_label="evening",
        concept_code="snomed:4136346",
        omop_id=3157002
    )
    insert_result = db.insert_mapping(
        variable_name="follow-up 24 months",
        standard_label="Follow-up 2 years",
        concept_code="snomed:199581000000104",
        omop_id=44789049
    )
    
    insert_result = db.insert_mapping(
        variable_name="36 months",
        standard_label="Follow-up 3 years",
        concept_code="icare:icv2000000004",
        omop_id=2000000004
    )
    insert_result = db.insert_mapping(
        variable_name="african",
        standard_label="african",
        concept_code="omop:3.03",
        omop_id=38003600
    )
    
    insert_result = db.insert_mapping(
        variable_name="RV pacemaker (VVI/DDDD)",
        standard_label="Cardiac pacemaker",
        concept_code="snomed:14106009",
        omop_id=4030875
    )

    insert_result = db.insert_mapping(
        variable_name="Cardiac pacemaker VVI",
        standard_label="Cardiac pacemaker",
        concept_code="snomed:14106009",
        omop_id=4030875
    )
    insert_result = db.insert_mapping(
        variable_name="Cardiac pacemaker DDDD",
        standard_label="Cardiac pacemaker",
        concept_code="snomed:14106009",
        omop_id=4030875
    )

    # for row in db.all_mappings():
    #     print(row)

    db.close_connection()
