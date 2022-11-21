
"""
Extract the documents from the PubMed Baseline dump (https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/)
"""

import sys
import json
import glob
import gzip
import pubmed_parser as pp

if __name__ == '__main__':

    with open(sys.argv[2], 'w') as o:

        for filename in sorted(glob.glob(sys.argv[1] + "/*.gz")):
            print(filename)
            with gzip.open(filename, 'r') as f:

                dicts_out = pp.parse_medline_xml(filename)
                papers = []
                for doc in dicts_out:
                    paper = {
                        "id": doc['pmid'],
                        "title": doc['title'].strip(),
                        "abstract": doc['abstract'].strip(),
                        "authors": [author.strip() for author in doc['authors'].strip().split(';') if author.strip()],
                        "keyphrases": [kw.strip() for kw in doc['keywords'].strip().split(';') if kw.strip()],
                        "mesh_terms": [mt.strip() for mt in doc['mesh_terms'].strip().split(';') if mt.strip()],
                        "year": doc['pubdate'].strip(),
                    }

                    # skip papers not in english
                    if not doc["languages"].startswith("eng"):
                        continue

                    # skip papers not having a title/abstract or keyphrases
                    if not paper["title"] or not paper["abstract"] or not len(paper["keyphrases"]):
                        continue
                    papers.append(paper)

                o.write("\n".join([json.dumps(p) for p in papers]))
