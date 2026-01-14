from __future__ import annotations

import json

try:  # allow use both as package import and standalone module
    from . import models  # type: ignore
except ImportError:  # pragma: no cover
    import models  # type: ignore


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


class CommitteeMeeting(models.Base):
    __tablename__ = "committee_meeting"

    id = models.Column(models.Integer, primary_key=True, autoincrement=True)
    meeting_num = models.Column(models.Integer, index=True)
    lthing = models.Column(models.Integer, index=True)
    nefnd_id = models.Column(models.Integer, index=True, nullable=True)
    start_time = models.Column(models.Text, nullable=True)
    end_time = models.Column(models.Text, nullable=True)
    raw_xml = models.Column(models.Text)

    __table_args__ = (
        models.UniqueConstraint("meeting_num", "lthing", name="uq_committee_meeting_num"),
    )


class CommitteeAttendance(models.Base):
    __tablename__ = "committee_attendance"

    id = models.Column(models.Integer, primary_key=True, autoincrement=True)
    meeting_id = models.Column(models.Integer, index=True)
    meeting_num = models.Column(models.Integer, index=True)
    lthing = models.Column(models.Integer, index=True)
    member_id = models.Column(models.Integer, index=True)
    status = models.Column(models.Text)  # present, proxy_present, notified_absent, absent
    role = models.Column(models.Text, nullable=True)
    substitute_for_member_id = models.Column(models.Integer, nullable=True)
    arrival_time = models.Column(models.Text, nullable=True)
    leave_note = models.Column(models.Text, nullable=True)

    __table_args__ = (
        models.UniqueConstraint("meeting_id", "member_id", name="uq_committee_attendance"),
    )


class VoteSession(models.Base):
    __tablename__ = "vote_session"

    id = models.Column(models.Integer, primary_key=True, autoincrement=True)
    lthing = models.Column(models.Integer, index=True)
    vote_num = models.Column(models.Integer, index=True)
    time = models.Column(models.Text, nullable=True)

    __table_args__ = (
        models.UniqueConstraint("lthing", "vote_num", name="uq_vote_session_vote_num"),
    )


class IssueMetrics(models.Base):
    __tablename__ = "issue_metrics"

    id = models.Column(models.Integer, primary_key=True, autoincrement=True)
    lthing = models.Column(models.Integer, index=True)
    malnr = models.Column(models.Integer, index=True)
    answer_status = models.Column(models.Text, index=True, nullable=True)  # svarað / ósvarað
    answer_latency = models.Column(models.Integer, nullable=True)  # business days

    __table_args__ = (
        models.UniqueConstraint("lthing", "malnr", name="uq_issue_metrics_malnr"),
    )


class Speech(models.Base):
    __tablename__ = "speech"

    id = models.Column(models.Integer, primary_key=True, autoincrement=True)
    lthing = models.Column(models.Integer, index=True)
    speech_id = models.Column(models.Text, index=True)
    member_id = models.Column(models.Integer, index=True, nullable=True)
    speaker_name = models.Column(models.Text, nullable=True)
    speaker_role = models.Column(models.Text, nullable=True)
    date = models.Column(models.Text, nullable=True)
    fundur = models.Column(models.Integer, nullable=True)
    fundarheiti = models.Column(models.Text, nullable=True)
    start_time = models.Column(models.Text, nullable=True)
    end_time = models.Column(models.Text, nullable=True)
    duration_seconds = models.Column(models.Integer, nullable=True)
    word_count = models.Column(models.Integer, nullable=True)
    words_per_minute = models.Column(models.Float, nullable=True)
    issue_malnr = models.Column(models.Integer, nullable=True)
    issue_malsflokkur = models.Column(models.Text, nullable=True)
    issue_malsheiti = models.Column(models.Text, nullable=True)
    kind = models.Column(models.Text, nullable=True)
    umraeda = models.Column(models.Text, nullable=True)
    audio_url = models.Column(models.Text, nullable=True)
    html_url = models.Column(models.Text, nullable=True)
    xml_url = models.Column(models.Text, nullable=True)
    raw_text = models.Column(models.Text, nullable=True)

    __table_args__ = (
        models.UniqueConstraint("lthing", "speech_id", name="uq_speech_lthing_id"),
    )
