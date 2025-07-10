import json
import sqlite3
from typing import Optional, List, Dict, Any

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
            self.insert_mapping(**mapping)

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
                        "variable_name": entry.get("variable_label"),
                        "standard_label": entry.get("standard_label"),
                        "concept_code": entry.get("code"),
                        "omop_id": entry.get("omop_id"),
                    })
            self.insert_many(unique_mappings)
            return {"status": "success", "message": f"Bulk insert completed. {len(unique_mappings)} unique rows inserted."}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def find_by_label(self, label: str) -> List[tuple]:
        # Substring, case-insensitive match
        pattern = f"%{label.lower()}%"
        self.cursor.execute(
            "SELECT * FROM concept_mappings WHERE LOWER(standard_label) LIKE ?",
            (pattern,)
        )
        results= self.cursor.fetchall()
        return results[0] if results else None

    def find_by_variable(self, variable_name: str) -> List[tuple]:
        # Substring, case-insensitive match for variable_name too
        pattern = f"%{variable_name.lower()}%"
        self.cursor.execute(
            "SELECT * FROM concept_mappings WHERE LOWER(variable_name) LIKE ?",
            (pattern,)
        )
        results = self.cursor.fetchall()
        return results[0] if results else None

    def all_mappings(self) -> List[tuple]:
        self.cursor.execute("SELECT * FROM concept_mappings")
        return self.cursor.fetchall()

    def close_connection(self):
        self.conn.close()


# Example usage
if __name__ == "__main__":
    db = DataManager("variables.db", initial_json="/Users/komalgilani/Desktop/cde_mapper/data/input/mapping_templates.json")

    # Insert bulk from JSON file
    # result = db.insert_mapping_bulk_json("/Users/komalgilani/Desktop/cde_mapper/data/input/mapping_templates.json")
    # print(result)

    # Example query
    print("Query by label='baseline':")
    print(db.find_by_variable("baseline"))

    print("Query by variable_name='month30_a':")
    print(db.find_by_variable("month30_a"))

    print("All mappings:")
    # for row in db.all_mappings():
    #     print(row)

    db.close()
