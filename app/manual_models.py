from __future__ import annotations

import json

from . import models


def attach_flutningsmenn(doc):
    """Parse and attach flutningsmenn JSON to _flutningsmenn attribute."""
    parsed_fm = []
    if getattr(doc, "leaf_kalladaftur", None):
        try:
            loaded = json.loads(doc.leaf_kalladaftur)
            if isinstance(loaded, list):
                parsed_fm = loaded
        except Exception:
            parsed_fm = []
    doc._flutningsmenn = parsed_fm


class IssueDocument(models.Base):
    __tablename__ = "issue_documents"

    id = models.Column(models.Integer, primary_key=True, autoincrement=True)
    lthing = models.Column(models.Integer, index=True)
    malnr = models.Column(models.Integer, index=True)
    malflokkur = models.Column(models.Text, index=True, nullable=True)
    skjalnr = models.Column(models.Integer, index=True)
    skjalategund = models.Column(models.Text)
    utbyting = models.Column(models.Text)
    slod_html = models.Column(models.Text)
    slod_pdf = models.Column(models.Text)
    slod_xml = models.Column(models.Text)

    __table_args__ = (
        models.UniqueConstraint("lthing", "malnr", "skjalnr", name="uq_issue_doc"),
    )


class VoteDetail(models.Base):
    __tablename__ = "vote_details"

    id = models.Column(models.Integer, primary_key=True, autoincrement=True)
    lthing = models.Column(models.Integer, index=True)
    vote_num = models.Column(models.Integer, index=True)
    parent_id = models.Column(models.Integer, index=True)
    voter_id = models.Column(models.Integer, index=True)
    voter_name = models.Column(models.Text)
    voter_xml = models.Column(models.Text)
    vote = models.Column(models.Text)

    __table_args__ = (
        models.UniqueConstraint("lthing", "vote_num", "voter_id", name="uq_vote_detail"),
    )


class MemberSeat(models.Base):
    __tablename__ = "member_seat"

    id = models.Column(models.Integer, primary_key=True, autoincrement=True)
    lthing = models.Column(models.Integer, index=True)
    member_id = models.Column(models.Integer, index=True)
    party_id = models.Column(models.Integer, index=True, nullable=True)
    party_name = models.Column(models.Text, nullable=True)
    type = models.Column(models.Text, nullable=True)
    kjordaemi_id = models.Column(models.Integer, nullable=True)
    kjordaemi_name = models.Column(models.Text, nullable=True)
    inn = models.Column(models.Text, nullable=True)
    ut = models.Column(models.Text, nullable=True)

    __table_args__ = (
        models.UniqueConstraint("lthing", "member_id", "party_id", "inn", name="uq_member_seat"),
    )


class NefndMember(models.Base):
    __tablename__ = "nefnd_member"

    id = models.Column(models.Integer, primary_key=True, autoincrement=True)
    lthing = models.Column(models.Integer, index=True)
    nefnd_id = models.Column(models.Integer, index=True)
    member_id = models.Column(models.Integer, index=True, nullable=True)
    name = models.Column(models.Text, nullable=True)
    role = models.Column(models.Text, nullable=True)
    inn = models.Column(models.Text, nullable=True)
    ut = models.Column(models.Text, nullable=True)

    __table_args__ = (
        models.UniqueConstraint("lthing", "nefnd_id", "member_id", "name", "inn", name="uq_nefnd_member"),
    )
