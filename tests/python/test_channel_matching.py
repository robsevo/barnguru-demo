"""
Sub-brand guard for the IPTV channel matcher (dashboard/api/main.py::_ch_matches).

Providers ship whole families of channels under one prefix. The matcher used to
accept any title starting with the curated name, so every sibling landed on the
parent: "MTV Classic"/"MTV 2"/"MTV Lebanon" all became MTV, "Lifetime Movie
Network" became Lifetime, "TSN The Ocho" (which is ESPN8) became TSN. Users saw it
as "MTV plays some Arab channel" and "this channel doesn't work" — the wrong-channel
sources rank next to the right ones, so which one you get is luck of the probe.

These tests pin both halves of the rule: a tail of pure feed qualifiers is the SAME
channel, a tail containing any other word is a DIFFERENT one.
"""
import pytest

from dashboard.api.main import _ch_matches, _normalize_ch


# (provider title, curated channel, should_match)
SUB_BRANDS_MUST_NOT_MATCH = [
    # The reported bug: MTV's pool was 164 candidates, ~20 of them actually MTV.
    ("US - MTV CLASSIC SD", "MTV"),
    ("CA - MTV 2 HD", "MTV"),
    ("USA - MTV U HD", "MTV"),
    ("CA - MTV DATING", "MTV"),
    ("CA - MTV REALITY", "MTV"),
    ("CA - MTV RIDICULOUSNESS", "MTV"),
    ("US - MTV Biggest Pop HD", "MTV"),
    ("US - MTV Spankin' New HD", "MTV"),
    ("UK - MTV LIVE HD", "MTV"),
    # Other channels the same defect was corrupting.
    ("USA - LIFETIME MOVIE NETWORK", "Lifetime"),
    ("US - DISCOVERY LIFE", "Discovery"),
    ("US - DISCOVERY FAMILY", "Discovery"),
    ("CA - TSN THE OCHO HD", "TSN"),          # this is ESPN8, not TSN
    ("US - ESPN SEC NETWORK HD", "ESPN"),
    ("US - ESPN ACC NETWORK HD", "ESPN"),
    ("CA - SPORTSNET WORLD HD", "Sportsnet"),
    ("CA - SPORTSNET PPV HD", "Sportsnet"),
    ("US - NAT GEO WILD", "National Geographic"),
    ("US - NESN PLUS", "NESN"),
    ("US - HBO 2 EASTERN FEED", "HBO"),
    ("US - CNBC WORLD", "CNBC"),
    ("CA - CTV DRAMA", "CTV"),                # a different Bell channel
    ("CA - CTV SCI-FI", "CTV"),
    ("CA - CBC NEWS", "CBC"),                 # CBC News Network, not CBC
    ("US - BET HER", "BET"),
    ("US - CINEMAX ACTION", "Cinemax"),
    ("US - STARZ IN BLACK", "Starz"),
    ("US - SHOWTIME FAMILY", "Showtime"),
    ("CA - BALLY SPORTS NORTH PLUS", "Bally Sports North"),
]

# Same channel, different feed — these must keep matching or the channel goes dark.
FEED_VARIANTS_MUST_MATCH = [
    ("US - MTV HD ◉", "MTV"),
    ("USA: MTV", "MTV"),
    ("MTV USA Eastern Feed HD (playlist)", "MTV"),
    ("USA MTV West", "MTV"),
    ("USA MTV East (SHD)", "MTV"),
    ("CA - SHOWCASE HD", "Showcase"),
    ("CA - SHOWCASE EAST", "Showcase"),
    ("US - LIFETIME HEVC", "Lifetime"),
    ("US - DISCOVERY CHANNEL HD", "Discovery"),
    ("US - HISTORY CHANNEL", "History"),
    ("US - CBS SPORTS NETWORK USA", "CBS Sports Network"),
    ("US - NFL REDZONE HD [BK]", "NFL RedZone"),
    ("CA - TSN 2 SD²", "TSN2"),
    ("US - NAT GEO WILD EAST", "Nat Geo Wild"),
]

# Canadian networks have no national feed in these pools — the local affiliate IS
# the channel, so the city tail has to be accepted for these names only.
CA_AFFILIATES_MUST_MATCH = [
    ("CA CBC Toronto", "CBC"),
    ("CA CBC Vancouver", "CBC"),
    ("CA| Citytv Toronto HD", "Citytv"),
    ("CA Citytv Montreal", "Citytv"),
    ("CA-FR TVA MONTREAL", "TVA"),
    ("CA: ICI TELE MONTREAL", "ICI Tele"),
    ("CA CTV Ottawa", "CTV"),
    ("CA CTV 2 Vancouver", "CTV 2"),
]

# Channels that pool their numbered sub-feeds ON PURPOSE (see the DAZN/beIN note in
# _LOUNGE_CHANNEL_NAMES): one channel over the union of feeds beats five flaky ones.
UNION_FEEDS_MUST_MATCH = [
    ("US - DAZN 1", "DAZN"),
    ("US - DAZN 4", "DAZN"),
    ("US - BEIN SPORTS 5", "beIN Sports"),
    ("US - MLS SOCCER 07", "MLS"),
    ("SERIE A 2 HD", "Serie A"),
    ("DE - SKY SPORT BUNDESLIGA 1", "Sky Sport Bundesliga"),
]


@pytest.mark.parametrize("title,channel", SUB_BRANDS_MUST_NOT_MATCH)
def test_sub_brand_does_not_claim_parent(title, channel):
    assert not _ch_matches(title, channel), (
        f"{title!r} (normalised {_normalize_ch(title)!r}) must not be served as {channel!r}"
    )


@pytest.mark.parametrize("title,channel", FEED_VARIANTS_MUST_MATCH + CA_AFFILIATES_MUST_MATCH
                         + UNION_FEEDS_MUST_MATCH)
def test_feed_variant_still_matches(title, channel):
    assert _ch_matches(title, channel), (
        f"{title!r} (normalised {_normalize_ch(title)!r}) is a feed of {channel!r} and must match"
    )


def test_numbered_feeds_only_pool_for_union_channels():
    """A digit tail is a feed for DAZN; for everyone else it is another channel."""
    assert _ch_matches("DAZN 2", "DAZN")
    assert not _ch_matches("MTV 2", "MTV")
    assert not _ch_matches("HBO 2", "HBO")
    assert not _ch_matches("NFL NETWORK 2", "NFL Network")


def test_short_french_match_is_word_bounded():
    """RDS matched as a bare substring, so "Drug LoRDS" and "WizaRDS" became RDS."""
    assert not _ch_matches("24/7: Drug Lords- The Takedown [42808]", "RDS")
    assert not _ch_matches("NBA Washington Wizards", "RDS")
    assert _ch_matches("CA: RDS (FR)", "RDS")
    assert _ch_matches("Canal RDS HD", "RDS")


def test_rds_2_claims_its_own_feeds():
    """Panels label it "CAFR - RDS 2 HD"; the CAFR tag survives normalisation, so
    RDS 2 needs the token match or its feeds fall through to plain RDS."""
    assert _ch_matches("CAFR - RDS 2 HD", "RDS 2")
    assert not _ch_matches("CAFR - RDS HD", "RDS 2")


def test_rebranded_regional_sports_networks_match():
    """Bally Sports North/Detroit became FanDuel Sports Network in 2025. The Bally
    titles are still listed but dead, which is why both channels had no working
    source at all."""
    assert _ch_matches("US - FANDUEL SPORTS NETWORK NORTH", "Bally Sports North")
    assert _ch_matches("US - FANDUEL SPORTS NETWORK DETROIT", "Bally Sports Detroit")
    assert not _ch_matches("US - FANDUEL SPORTS NETWORK FLORIDA", "Bally Sports North")
    # …and the national FanDuel channel is still its own thing.
    assert _ch_matches("US - FANDUEL TV HD", "FanDuel")
    assert not _ch_matches("US - FANDUEL RACING", "FanDuel")


def test_espn_plus_event_feeds_are_not_the_linear_channel():
    assert _ch_matches("ESPN Plus", "ESPN+")
    assert not _ch_matches(
        "ESPN+ 01 : Surfing – 2026 WSL Longboard Tour Jul 27 – 6:30 AM ET", "ESPN+")
