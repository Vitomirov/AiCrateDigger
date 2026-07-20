"""Host and path pattern constants for Tavily pre-filtering."""

from __future__ import annotations

import re

#: Hosts (or substrings) we never want to send to the LLM. Pure noise filter —
#: zero shop domains are listed here. Match is a case-insensitive substring on
#: the registrable host so subdomains (e.g. ``music.youtube.com``) are caught.
BLACKLIST_HOST_SUBSTRINGS: tuple[str, ...] = (
    # Video / streaming / lyrics
    "youtube.com",
    "youtu.be",
    "spotify.com",
    "music.apple.com",
    "soundcloud.com",
    "deezer.com",
    "tidal.com",
    "pandora.com",
    "bandcamp.com",
    "genius.com",
    "azlyrics.com",
    "lyrics.com",
    "songkick.com",
    "setlist.fm",
    # Encyclopedia / metadata
    "wikipedia.org",
    "wikidata.org",
    "musicbrainz.org",
    "allmusic.com",
    "rateyourmusic.com",
    "last.fm",
    "lastfm.",
    "imdb.com",
    "fandom.com",
    "rollingstone.com",
    "pitchfork.com",
    "stereogum.com",
    "factmag.com",
    # Social
    "facebook.com",
    "instagram.com",
    "twitter.com",
    "x.com",
    "tiktok.com",
    "reddit.com",
    "pinterest.com",
    "tumblr.com",
    "threads.net",
    "linkedin.com",
    "vk.com",
    # News / aggregators / search engines
    "news.google.",
    "bing.com",
    "duckduckgo.com",
    "yahoo.com",
    "msn.com",
    "tripadvisor.",
    "yelp.",
    "timeout.com",
    # Marketplace mega-shops (their localised TLD variants too — opportunistic
    # store discovery skips these so we never spend an LLM call verifying that
    # eBay-Germany is not an indie record shop).
    "ebay.com",
    "ebay.de",
    "ebay.co.uk",
    "ebay.it",
    "ebay.fr",
    "ebay.es",
    "ebay.nl",
    "ebay.at",
    "ebay.pl",
    "ebay.ie",
    "ebay.com.au",
    "amazon.com",
    "amazon.de",
    "amazon.co.uk",
    "amazon.it",
    "amazon.fr",
    "amazon.es",
    "amazon.nl",
    "amazon.pl",
    "amazon.com.au",
    "etsy.com",
    "allegro.pl",
    "avito.ru",
    # Discogs metadata (the public site is a hub without consistent buy-now PDP rows for our shop-pool intent).
    # without consistent buy-now PDP rows for our shop-pool intent).
    "discogs.com",
    "vinylhub.com",
    # Document hosts / archives that occasionally show up in SERPs
    "archive.org",
    "scribd.com",
    "medium.com",
    "boilerroom.tv",
    "ra.co",
)

#: Print-on-demand / band-merch / poster marketplaces. These sell apparel,
#: posters, pins and gift items branded with an artist's name — never a
#: buyable physical album — so an artist-name match alone (e.g. "Mgła Band
#: Merch & Gifts for Sale") must never survive the noise gate. Trailing-dot
#: entries cover country-TLD storefronts of the same platform (e.g.
#: ``spreadshirt.de``, ``spreadshirt.co.uk``).
MERCH_HOST_SUBSTRINGS: tuple[str, ...] = (
    "redbubble.com",
    "teepublic.com",
    "spreadshirt.",
    "zazzle.com",
    "society6.com",
    "displate.com",
    "fineartamerica.com",
    "cafepress.com",
    "threadless.com",
    "customink.com",
    "printful.com",
    "printify.com",
    "vograce.com",
    "gearbubble.com",
    "represent.com",
    "bonfire.com",
    "stickermule.com",
    "allposters.com",
    "desenio.com",
    "merchbar.com",
    "indiemerchstore.com",
    "districtlines.com",
    "hottopic.com",
    "spencersonline.com",
)

#: Digital-only download / streaming stores. No physical product ever ships
#: from these, so a title/artist match here can never satisfy a "buyable
#: physical copy" intent regardless of format wording in the snippet.
#: (Bandcamp/Spotify/Deezer/Tidal/Pandora/Apple Music already live in the
#: main blacklist above — this tuple covers the DJ/electronic-focused
#: digital retailers that would otherwise slip through.)
DIGITAL_MUSIC_HOST_SUBSTRINGS: tuple[str, ...] = (
    "beatport.com",
    "junodownload.com",
    "traxsource.com",
    "7digital.com",
    "qobuz.com",
    "hdtracks.com",
    "napster.com",
    "anghami.com",
    "boomplay.com",
    "audiomack.com",
)

#: Concert/event ticketing platforms — never sell physical albums, but
#: frequently outrank shops for "{artist} {country}" style queries.
EVENT_TICKETING_HOST_SUBSTRINGS: tuple[str, ...] = (
    "ticketmaster.",
    "eventbrite.",
    "livenation.com",
    "seetickets.",
    "bandsintown.com",
    "viagogo.",
    "stubhub.com",
)

#: News-portal / general media substrings — substring match anywhere in the
#: registrable host. Catches large European news outlets that publish "best
#: albums of 2026" listicles and surface in Tavily SERPs without ever selling
#: vinyl. *Negative* patterns only — no shop domains here.
NEWS_HOST_SUBSTRINGS: tuple[str, ...] = (
    # Generic news / info host words (multi-language)
    "blic.",
    "n1info.",
    "telegraf.",
    "kurir.",
    "novosti.",
    "espreso.",
    "danas.rs",
    "b92.",
    "rts.rs",
    "rtl.",
    "vesti.",
    "24sata.",
    "jutarnji.",
    "vecernji.",
    "index.hr",
    "hrt.hr",
    "dnevnik.",
    "tportal.",
    "spiegel.",
    "welt.",
    "faz.net",
    "sueddeutsche.",
    "bild.",
    "tagesschau.",
    "tagesspiegel.",
    "zeit.de",
    "stern.de",
    "focus.de",
    "lemonde.",
    "lefigaro.",
    "liberation.",
    "leparisien.",
    "lepoint.",
    "lexpress.",
    "ouest-france.",
    "bbc.co",
    "bbc.com",
    "theguardian.",
    "telegraph.co",
    "independent.co",
    "thetimes.",
    "dailymail.",
    "express.co",
    "mirror.co",
    "cnn.com",
    "nytimes.",
    "washingtonpost.",
    "reuters.",
    "apnews.",
    "aljazeera.",
    "euronews.",
    "rt.com",
    "tass.",
    "ria.ru",
    "corriere.",
    "repubblica.",
    "lastampa.",
    "gazzetta.",
    "elpais.",
    "elmundo.",
    "abc.es",
    "marca.com",
    "ansa.it",
    "publico.",
    "expresso.",
    "rtp.pt",
    "wp.pl",
    "onet.pl",
    "interia.pl",
    "tvn24.pl",
    "polskieradio.",
    "nu.nl",
    "telegraaf.nl",
    "ad.nl",
    "nrc.nl",
    "rte.ie",
    "ert.gr",
    "kathimerini.",
    "tovima.",
    "iefimerida.",
    "sport-klub.",
    "sportklub.",
    "hurriyet.",
    "milliyet.",
)

#: Path fragments that are almost always editorial / non-PDP even on legit
#: shop hosts (e.g. a record store's blog). Used to demote in scoring AND to
#: hard-reject *unknown* hosts that show no PDP signal.
EDITORIAL_PATH_SUBSTRINGS: tuple[str, ...] = (
    "/blog",
    "/news",
    "/article",
    "/articles",
    "/clanak/",
    "/clanci/",
    "/vesti/",
    "/magazine",
    "/feature",
    "/features",
    "/story",
    "/stories",
    "/playlist",
    "/playlists",
    "/podcast",
    "/podcasts",
    "/tag/",
    "/tags/",
    "/forum/",
    "/community/",
)

#: Path fragments that mean "band merch / apparel / event ticket", NOT a
#: buyable album. Unlike ``EDITORIAL_PATH_SUBSTRINGS`` (soft score demotion),
#: these are HARD-rejected for every host — including whitelisted shops —
#: because a legitimate record store's own ``/merch/`` or ``/apparel/``
#: category page is still not a physical album listing. See
#: :func:`app.domains.engine.search.prefilter.filter.prefilter_tavily_results`.
MERCH_PATH_SUBSTRINGS: tuple[str, ...] = (
    "/merch",
    "/merchandise",
    "/apparel",
    "/clothing",
    "/t-shirt",
    "/t-shirts",
    "/tshirt",
    "/tshirts",
    "/hoodie",
    "/hoodies",
    "/posters/",
    "/sticker",
    "/stickers",
    "/patches",
    "/enamel-pin",
    "/tote-bag",
    "/phone-case",
    "/giftshop",
    "/gift-shop",
    "/tickets/",
    "/ticket/",
)

WWW_PREFIX_RE = re.compile(r"^www\.", re.IGNORECASE)
