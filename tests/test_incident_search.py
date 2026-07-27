from datetime import datetime, timezone, timedelta
from backend.services.incident_search import (
    is_stale_article,
    parse_rss_items_for_incident,
    parse_pubdate,
    title_allows_incident_extraction,
    is_valid_incident_candidate,
)


def _fresh_pubdate(days_ago: int = 1) -> str:
    now = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return now.strftime("%a, %d %b %Y %H:%M:%S GMT")


def test_parse_pubdate_valid_rfc822():
    pubdate_str = "Wed, 27 Jul 2026 14:30:00 GMT"
    dt = parse_pubdate(pubdate_str)
    assert dt is not None
    assert dt.year == 2026
    assert dt.month == 7
    assert dt.day == 27


def test_is_stale_article_rejects_may_article_in_july():
    may_pubdate = "Wed, 15 May 2026 10:00:00 GMT"
    assert is_stale_article(may_pubdate, max_days=7) is True


def test_is_stale_article_accepts_fresh_article():
    assert is_stale_article(_fresh_pubdate(2), max_days=7) is False


def test_is_stale_article_rejects_missing_pubdate():
    assert is_stale_article(None, max_days=7) is True


def test_title_gate_blocks_urban_person_structure_fires():
    assert title_allows_incident_extraction("Man Starts Fire in Brooklyn vacant lot") is False
    assert title_allows_incident_extraction("House Fire destroys garage in Chicago suburb") is False
    assert title_allows_incident_extraction("Car Fire shuts down interstate near Dallas") is False
    assert title_allows_incident_extraction("Teen Sparks Fire behind school") is False


def test_title_gate_allows_named_wildfire_headlines():
    assert title_allows_incident_extraction("Creek Fire forces evacuations near ridge") is True
    assert title_allows_incident_extraction("Dixie Fire grows to 2000 acres") is True
    # Named incident without acres/evac still allowed if not urban pattern
    assert title_allows_incident_extraction("Creek Fire grows overnight in northern California") is True


def test_candidate_validation_is_structural():
    assert is_valid_incident_candidate("Creek") is True
    assert is_valid_incident_candidate("Hay Creek") is True
    assert is_valid_incident_candidate("Dixie") is True
    assert is_valid_incident_candidate("Brush") is False
    assert is_valid_incident_candidate("Structure") is False
    assert is_valid_incident_candidate("Man Starts") is False
    assert is_valid_incident_candidate("Teen Sparks") is False


def test_parse_rss_items_for_incident_filters_stale_may_fire():
    stale_rss_xml = """
    <rss version="2.0">
      <channel>
        <item>
          <title>Sandy Fire breaks out in southern forests after evacuations ordered</title>
          <pubDate>Wed, 15 May 2026 10:00:00 GMT</pubDate>
        </item>
      </channel>
    </rss>
    """
    assert parse_rss_items_for_incident(stale_rss_xml, max_days=7) is None


def test_parse_rss_items_for_incident_extracts_fresh_fire():
    fresh_rss_xml = f"""
    <rss version="2.0">
      <channel>
        <item>
          <title>Creek Fire forces evacuations near mountain ridge - Daily Press</title>
          <pubDate>{_fresh_pubdate()}</pubDate>
        </item>
      </channel>
    </rss>
    """
    assert parse_rss_items_for_incident(fresh_rss_xml, max_days=7) == "Creek Fire"


def test_rejects_urban_fire_headlines_any_us_city():
    junk_rss = f"""
    <rss version="2.0">
      <channel>
        <item>
          <title>Man Starts Fire in Brooklyn vacant lot - Local News</title>
          <pubDate>{_fresh_pubdate()}</pubDate>
        </item>
        <item>
          <title>House Fire destroys garage in Chicago suburb</title>
          <pubDate>{_fresh_pubdate()}</pubDate>
        </item>
        <item>
          <title>Creek Fire forces evacuations near mountain ridge after burning 800 acres</title>
          <pubDate>{_fresh_pubdate()}</pubDate>
        </item>
      </channel>
    </rss>
    """
    assert parse_rss_items_for_incident(junk_rss, max_days=7) == "Creek Fire"


def test_rejects_fire_type_generic_even_with_wildland_context():
    rss = f"""
    <rss version="2.0">
      <channel>
        <item>
          <title>Brush Fire burns 40 acres near highway, no evacuations ordered</title>
          <pubDate>{_fresh_pubdate()}</pubDate>
        </item>
      </channel>
    </rss>
    """
    assert parse_rss_items_for_incident(rss, max_days=7) is None
