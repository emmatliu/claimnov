import json, os, re, sys, urllib.request
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "iclr_novelty_validation_data")
PDFDIR = os.path.join(HERE, "scratch_pdfs")
GROBID = "http://localhost:8070/api/processReferences"
NS = {"t": "http://www.tei-c.org/ns/1.0"}
TARGETS = [("2O_pIShVl-", "2022"), ("f8zhJ1jAfIq", "2022")]


def grobid_refs(pdf_path):
    import requests
    with open(pdf_path, "rb") as f:
        r = requests.post(GROBID, files={"input": f},
                          data={"includeRawCitations": "1"}, timeout=180)
    r.raise_for_status()
    return r.text


def txt(el):
    return " ".join("".join(el.itertext()).split()) if el is not None else ""


def parse_tei(tei):
    root = ET.fromstring(tei.encode("utf-8"))
    out = []
    for bs in root.iter("{http://www.tei-c.org/ns/1.0}biblStruct"):
        title = bs.find(".//t:analytic/t:title", NS)
        if title is None:
            title = bs.find(".//t:monogr/t:title", NS)
        authors = []
        for pers in bs.findall(".//t:author/t:persName", NS):
            fore = " ".join(txt(f) for f in pers.findall("t:forename", NS))
            sur = txt(pers.find("t:surname", NS))
            name = (fore + " " + sur).strip()
            if name:
                authors.append(name)
        year = ""
        d = bs.find(".//t:date", NS)
        if d is not None:
            y = re.search(r"(19|20)\d{2}", (d.get("when") or txt(d) or ""))
            if y:
                year = y.group()
        raw = bs.find(".//t:note[@type='raw_reference']", NS)
        out.append({"title": txt(title), "authors": authors, "year": year,
                    "raw": txt(raw)})
    return out


def main():
    ok = []
    for aid, year in TARGETS:
        pdf = os.path.join(PDFDIR, aid + ".pdf")
        if not os.path.exists(pdf) or open(pdf, "rb").read(4) != b"%PDF":
            print(f"!! {aid}: PDF missing or not a real PDF at {pdf} — download it from "
                  f"https://openreview.net/pdf?id={aid} into scratch_pdfs/", file=sys.stderr)
            continue
        refs = parse_tei(grobid_refs(pdf))
        outdir = os.path.join(DATA, "refs", year)
        os.makedirs(outdir, exist_ok=True)
        outpath = os.path.join(outdir, aid + ".json")
        json.dump(refs, open(outpath, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"OK {aid}: {len(refs)} references -> refs/{year}/{aid}.json")
        ok.append(aid)
    print(f"\nparsed {len(ok)}/{len(TARGETS)}")


if __name__ == "__main__":
    main()
