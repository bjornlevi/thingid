import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sqlalchemy import Column, Integer, MetaData, Table, Text, UniqueConstraint, create_engine, text
from sqlalchemy.orm import declarative_base

from scripts.get_data import unique_constraint_columns
from scripts import get_data
from app import models


class IngestConstraintsTest(unittest.TestCase):
    def test_import_skips_duplicate_issue_with_database_only_constraint(self):
        resource = next(
            r for r in json.loads((get_data.ROOT_DIR / "schema_map.json").read_text())["resources"]
            if r["table"] == "thingmalalisti__mal"
        )
        xml = (
            '<málalisti>'
            '<mál málsnúmer="38" þingnúmer="24" málsflokkur="A"/>'
            '<mál málsnúmer="38" þingnúmer="24" málsflokkur="A"/>'
            '<mál málsnúmer="38" þingnúmer="24" málsflokkur="B"/>'
            '</málalisti>'
        ).encode()
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "test.db"
            schema = Path(directory) / "schema.json"
            schema.write_text(json.dumps({"resources": [resource]}))
            metadata = MetaData()
            table = models.ThingmalalistiMal.__table__.to_metadata(metadata)
            table.append_constraint(UniqueConstraint(
                "attr_malsnumer", "attr_malsflokkur", "ingest_lthing", "ingest_resource",
            ))
            engine = create_engine(f"sqlite:///{db}")
            try:
                metadata.create_all(engine)
                output = io.StringIO()
                with (
                    patch("sys.argv", [
                        "get_data.py", "--db", str(db), "--schema", str(schema),
                        "--lthing", "24", "--skip-speeches",
                        "--cache-dir", str(Path(directory) / "cache"),
                    ]),
                    patch.object(get_data.importlib, "import_module",
                                 side_effect=lambda name: models if name == "models" else None),
                    patch.object(get_data, "discover_current_lthing_and_yfirlit", return_value=(157, {})),
                    patch.object(get_data, "discover_lthing_and_yfirlit", return_value=(24, {})),
                    patch.object(get_data.Fetcher, "get", return_value=xml),
                    patch.object(get_data, "cache_nefndarfundir"),
                    contextlib.redirect_stdout(output),
                ):
                    self.assertEqual(get_data.main(), 0)
                self.assertIn("1 skipped as duplicates", output.getvalue())
                with engine.connect() as connection:
                    self.assertEqual(connection.execute(text(
                        "SELECT attr_malsflokkur FROM thingmalalisti__mal ORDER BY attr_malsflokkur"
                    )).scalars().all(), ["A", "B"])
            finally:
                engine.dispose()

    def test_existing_database_constraint_missing_from_model(self):
        base = declarative_base()

        class Issue(base):
            __tablename__ = "issues"
            id = Column(Integer, primary_key=True)
            number = Column(Integer)
            category = Column(Text)
            lthing = Column(Integer)
            resource = Column(Text)

        metadata = MetaData()
        Table(
            "issues",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("number", Integer),
            Column("category", Text),
            Column("lthing", Integer),
            Column("resource", Text),
            UniqueConstraint("number", "category", "lthing", "resource"),
        )
        engine = create_engine("sqlite://")
        try:
            metadata.create_all(engine)
            with engine.connect() as connection:
                self.assertEqual(
                    unique_constraint_columns(Issue, connection),
                    [["number", "category", "lthing", "resource"]],
                )
        finally:
            engine.dispose()

    def test_matching_model_and_database_constraints_are_not_duplicated(self):
        base = declarative_base()

        class Member(base):
            __tablename__ = "members"
            id = Column(Integer, primary_key=True)
            number = Column(Integer)
            __table_args__ = (UniqueConstraint("number"),)

        engine = create_engine("sqlite://")
        try:
            base.metadata.create_all(engine)
            with engine.connect() as connection:
                self.assertEqual(unique_constraint_columns(Member, connection), [["number"]])
        finally:
            engine.dispose()

    def test_table_without_unique_constraints(self):
        base = declarative_base()

        class Vote(base):
            __tablename__ = "votes"
            id = Column(Integer, primary_key=True)

        engine = create_engine("sqlite://")
        try:
            base.metadata.create_all(engine)
            with engine.connect() as connection:
                self.assertEqual(unique_constraint_columns(Vote, connection), [])
        finally:
            engine.dispose()


if __name__ == "__main__":
    unittest.main()
