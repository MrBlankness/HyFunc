import json
import re
import struct
import unittest
import xml.etree.ElementTree as ET
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "docs"


class ProjectPage(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.elements = []
        self.metadata = {}
        self.authors = []
        self.structured_data = []
        self.citation = []
        self.paper_title = []
        self._capture = None
        self.feed(source)

    def handle_starttag(self, tag, attributes):
        attributes = dict(attributes)
        self.elements.append((tag, attributes))
        if tag == "meta":
            name = attributes.get("name", attributes.get("property"))
            self.metadata[name] = attributes.get("content")
            if name == "citation_author":
                self.authors.append(attributes["content"])
        if tag == "script" and attributes.get("type") == "application/ld+json":
            self._capture = ("script", self.structured_data)
        elif attributes.get("id") == "bibtex":
            self._capture = ("code", self.citation)
        elif attributes.get("id") == "paper-title":
            self._capture = (tag, self.paper_title)

    def handle_data(self, data):
        if self._capture:
            self._capture[1].append(data)

    def handle_endtag(self, tag):
        if self._capture and tag == self._capture[0]:
            self._capture = None


class ProjectSiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.page = ProjectPage((SITE / "index.html").read_text())
        cls.graph = json.loads("".join(cls.page.structured_data))["@graph"]
        cls.entities = {entity["@type"]: entity for entity in cls.graph}
        cls.canonical = next(
            attributes["href"]
            for tag, attributes in cls.page.elements
            if tag == "link" and attributes.get("rel") == "canonical"
        )

    def test_structured_entities_and_social_preview_are_connected(self):
        website = self.entities["WebSite"]
        webpage = self.entities["WebPage"]
        article = self.entities["ScholarlyArticle"]
        software = self.entities["SoftwareSourceCode"]
        self.assertEqual(website["url"], self.canonical)
        self.assertEqual(webpage["url"], self.canonical)
        self.assertEqual(webpage["isPartOf"]["@id"], website["@id"])
        self.assertEqual(webpage["mainEntity"]["@id"], article["@id"])
        self.assertEqual(webpage["about"]["@id"], software["@id"])
        self.assertEqual(article["mainEntityOfPage"]["@id"], webpage["@id"])
        self.assertEqual(software["citation"]["@id"], article["@id"])

        sitemap = ET.parse(SITE / "sitemap.xml")
        namespace = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        self.assertEqual(sitemap.findtext(".//s:loc", namespaces=namespace), self.canonical)

        image = webpage["primaryImageOfPage"]
        self.assertEqual(image["url"], self.page.metadata["og:image"])
        self.assertEqual(image["url"], self.page.metadata["twitter:image"])
        self.assertTrue(image["url"].startswith(self.canonical))
        image_path = SITE / unquote(image["url"][len(self.canonical):])
        png = image_path.read_bytes()
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        width, height = struct.unpack(">II", png[16:24])
        self.assertEqual((width, height), (image["width"], image["height"]))
        self.assertEqual(str(width), self.page.metadata["og:image:width"])
        self.assertEqual(str(height), self.page.metadata["og:image:height"])

    def test_visible_and_machine_readable_citations_match_bibtex(self):
        bibtex = (ROOT / "CITATION.bib").read_text()
        title = re.search(r"title\s*=\s*\{(.*?)\},\s*booktitle", bibtex, re.S)[1]
        title = " ".join(title.replace("{", "").replace("}", "").split())
        authors = re.search(r"author\s*=\s*\{(.*?)\}", bibtex)[1].split(" and ")
        doi = re.search(r"doi\s*=\s*\{(.*?)\}", bibtex)[1]
        article = self.entities["ScholarlyArticle"]
        self.assertEqual(article["name"], title)
        self.assertEqual(self.page.metadata["citation_title"], title)
        self.assertEqual(" ".join("".join(self.page.paper_title).split()), title)
        self.assertEqual([author["name"] for author in article["author"]], authors)
        self.assertEqual(self.page.authors, authors)
        self.assertEqual(article["identifier"], doi)
        self.assertEqual(self.page.metadata["citation_doi"], doi)
        self.assertEqual("".join(self.page.citation).strip(), bibtex.strip())

    def test_static_assets_and_fragment_links_resolve_without_a_build(self):
        identifiers = [
            attributes["id"] for _, attributes in self.page.elements if "id" in attributes
        ]
        self.assertEqual([name for name, count in Counter(identifiers).items() if count > 1], [])
        self.assertEqual(sum(tag == "h1" for tag, _ in self.page.elements), 1)
        for tag, attributes in self.page.elements:
            with self.subTest(tag=tag, attributes=attributes):
                href = attributes.get("href", "")
                if href.startswith("#"):
                    self.assertIn(href[1:], identifiers)
                reference = attributes.get("src")
                if tag == "link" and attributes.get("rel") in {"stylesheet", "icon", "alternate"}:
                    reference = href
                if reference:
                    self.assertFalse(urlsplit(reference).scheme, "Assets must be self-hosted")
                    self.assertTrue((SITE / unquote(reference)).is_file(), reference)
                if tag == "img":
                    self.assertTrue(attributes.get("alt", "").strip())
                    self.assertGreater(int(attributes["width"]), 0)
                    self.assertGreater(int(attributes["height"]), 0)

        svg = ET.parse(SITE / "assets/architecture.svg").getroot()
        labelled_by = svg.attrib["aria-labelledby"].split()
        for identifier in labelled_by:
            label = next(element for element in svg.iter() if element.attrib.get("id") == identifier)
            self.assertTrue("".join(label.itertext()).strip())
        for asset in (SITE / "assets").glob("*.svg"):
            ET.parse(asset)


if __name__ == "__main__":
    unittest.main()
