import requests
from bs4 import BeautifulSoup
from bs4 import FeatureNotFound
import os
import re
import time
from tqdm import tqdm

from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

ELSEVIER_API_KEY = os.getenv("ELSEVIER_API_KEY") or os.getenv("ELSEVIER_API")

METHOD_SECTION_PATTERN = re.compile(
    r"\b(methodology|methods?|materials\s*(?:and|&)\s*methods?|experimental methods?|experimental)\b",
    re.IGNORECASE,
)


def _normalize_text(text):
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _tag_text(tag):
    if tag is None:
        return ""
    return _normalize_text(tag.get_text(" ", strip=True))


def _parse_xml_soup(xml_input):
    """
    Parse XML into BeautifulSoup using XML parsers only.
    """
    parsers = ["lxml-xml", "xml"]
    last_exc = None
    for parser in parsers:
        try:
            return BeautifulSoup(xml_input, parser)
        except FeatureNotFound as exc:
            last_exc = exc
            continue

    raise RuntimeError(
        "Could not parse XML with available XML parsers. "
        "Install lxml for best support: pip install lxml"
    ) from last_exc


def fetch_elsevier_xml(doi, api_key=None, timeout=30):
    """
    Fetch XML document for a given DOI from the Elsevier API.
    
    Parameters:
    -----------
    doi : str
        Digital Object Identifier for the article
        
    Returns:
    --------
    BeautifulSoup object or None
        BeautifulSoup object of the XML document if successful, None otherwise
    """
    if not api_key:
        # if no api key provided
        api_key = ELSEVIER_API_KEY
    if not api_key:
        print(f"Missing Elsevier API key; skipping DOI {doi}")
        return None
    url = f'https://api.elsevier.com/content/article/doi/{doi}?view=FULL&APIKey={api_key}'
    
    headers = {
        'Accept': 'application/xml'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Parse as XML only; do not fall back to HTML parsers.
        soup = _parse_xml_soup(response.content)
        
        return soup
    except RuntimeError as e:
        print(f"XML Parser Error for DOI {doi}: {e}")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error for DOI {doi}: {e} (Status Code: {e.response.status_code})")
        return None
    except requests.exceptions.ConnectionError as e:
        print(f"Connection Error for DOI {doi}: {e}")
        return None
    except requests.exceptions.Timeout as e:
        print(f"Timeout Error for DOI {doi}: {e}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching XML for DOI {doi}: {e}")
        return None


def extract_structured_sections(xml_input):
    """
    Extract section-aware textual content from an Elsevier XML document.

    Parameters:
    -----------
    xml_input : BeautifulSoup | str | bytes
        XML content as a BeautifulSoup object or raw XML string/bytes.

    Returns:
    --------
    list[dict]
        Ordered section blocks with keys:
        - section_title: str
        - section_type: str
        - text: str
    """
    if xml_input is None:
        return []

    soup = xml_input if isinstance(xml_input, BeautifulSoup) else _parse_xml_soup(xml_input)

    blocks = []

    title_tag = soup.find("dc:title") or soup.find("ce:title") or soup.find("title")
    title_text = _tag_text(title_tag)
    if title_text:
        blocks.append(
            {
                "section_title": "Title",
                "section_type": "title",
                "text": title_text,
            }
        )

    abstract_nodes = soup.find_all(["ce:abstract", "abstract", "dc:description"])
    for node in abstract_nodes:
        abstract_text = _tag_text(node)
        if abstract_text:
            blocks.append(
                {
                    "section_title": "Abstract",
                    "section_type": "abstract",
                    "text": abstract_text,
                }
            )

    section_nodes = soup.find_all(["ce:section", "section"])
    for node in section_nodes:
        section_title_tag = (
            node.find("ce:section-title")
            or node.find("section-title")
            or node.find("title")
            or node.find("head")
        )
        section_title = _tag_text(section_title_tag) or "Body"

        paragraph_nodes = node.find_all(["ce:para", "para", "p"])
        paragraph_texts = [_tag_text(p) for p in paragraph_nodes]
        paragraph_texts = [p for p in paragraph_texts if p]

        if paragraph_texts:
            blocks.append(
                {
                    "section_title": section_title,
                    "section_type": "body",
                    "text": "\n\n".join(paragraph_texts),
                }
            )

    if not section_nodes:
        body_node = soup.find("ce:body") or soup.find("body")
        if body_node:
            paragraph_nodes = body_node.find_all(["ce:para", "para", "p"])
            paragraph_texts = [_tag_text(p) for p in paragraph_nodes]
            paragraph_texts = [p for p in paragraph_texts if p]
            if paragraph_texts:
                blocks.append(
                    {
                        "section_title": "Body",
                        "section_type": "body",
                        "text": "\n\n".join(paragraph_texts),
                    }
                )

    caption_nodes = soup.find_all(["ce:caption", "caption"])
    for node in caption_nodes:
        caption_text = _tag_text(node)
        if caption_text:
            blocks.append(
                {
                    "section_title": "Caption",
                    "section_type": "caption",
                    "text": caption_text,
                }
            )

    # Preserve order but de-duplicate repeated chunks.
    seen = set()
    deduped_blocks = []
    for block in blocks:
        key = (block["section_title"], block["section_type"], block["text"])
        if key in seen:
            continue
        seen.add(key)
        deduped_blocks.append(block)

    return deduped_blocks


def extract_plain_text(xml_input):
    """
    Convert XML into a section-marked plain-text document suitable for chunking.
    """
    sections = extract_structured_sections(xml_input)
    lines = []
    for section in sections:
        title = section.get("section_title", "Section")
        text = section.get("text", "")
        if not text:
            continue
        lines.append(f"## {title}\n{text}")
    return "\n\n".join(lines)


def extract_methodology_sections(xml_input):
    """
    Extract only methodology-related sections from an Elsevier XML document.

    Parameters:
    -----------
    xml_input : BeautifulSoup | str | bytes
        XML content as a BeautifulSoup object or raw XML string/bytes.

    Returns:
    --------
    list[dict]
        Ordered methodology blocks with keys:
        - section_title: str
        - section_type: str
        - text: str
    """
    if xml_input is None:
        return []

    soup = xml_input if isinstance(xml_input, BeautifulSoup) else _parse_xml_soup(xml_input)

    blocks = []
    section_nodes = soup.find_all(["ce:section", "section"])

    for node in section_nodes:
        section_title_tag = (
            node.find("ce:section-title")
            or node.find("section-title")
            or node.find("title")
            or node.find("head")
        )
        section_title = _tag_text(section_title_tag)
        if not section_title or not METHOD_SECTION_PATTERN.search(section_title):
            continue

        paragraph_nodes = node.find_all(["ce:para", "para", "p"])
        paragraph_texts = [_tag_text(p) for p in paragraph_nodes]
        paragraph_texts = [p for p in paragraph_texts if p]

        if paragraph_texts:
            blocks.append(
                {
                    "section_title": section_title,
                    "section_type": "methodology",
                    "text": "\n\n".join(paragraph_texts),
                }
            )

    if not blocks:
        body_node = soup.find("ce:body") or soup.find("body")
        if body_node:
            section_candidates = body_node.find_all(["ce:section", "section"])
            for node in section_candidates:
                section_title_tag = (
                    node.find("ce:section-title")
                    or node.find("section-title")
                    or node.find("title")
                    or node.find("head")
                )
                section_title = _tag_text(section_title_tag)
                if not section_title or not METHOD_SECTION_PATTERN.search(section_title):
                    continue

                paragraph_nodes = node.find_all(["ce:para", "para", "p"])
                paragraph_texts = [_tag_text(p) for p in paragraph_nodes]
                paragraph_texts = [p for p in paragraph_texts if p]

                if paragraph_texts:
                    blocks.append(
                        {
                            "section_title": section_title,
                            "section_type": "methodology",
                            "text": "\n\n".join(paragraph_texts),
                        }
                    )

    seen = set()
    deduped_blocks = []
    for block in blocks:
        key = (block["section_title"], block["section_type"], block["text"])
        if key in seen:
            continue
        seen.add(key)
        deduped_blocks.append(block)

    return deduped_blocks


def extract_methodology_text(xml_input):
    """
    Convert only methodology-related XML sections into plain text.
    """
    sections = extract_methodology_sections(xml_input)
    lines = []
    for section in sections:
        title = section.get("section_title", "Methodology")
        text = section.get("text", "")
        if not text:
            continue
        lines.append(f"## {title}\n{text}")
    return "\n\n".join(lines)

def save_xml_to_file(doi, xml_soup, output_dir="elsevier_xml"):
    """
    Save XML content to a file.
    
    Parameters:
    -----------
    doi : str
        Digital Object Identifier for the article
    xml_soup : BeautifulSoup object
        BeautifulSoup object of the XML document
    output_dir : str
        Directory where XML files will be saved
        
    Returns:
    --------
    str or None
        Path to the saved file if successful, None otherwise
    """
    if not xml_soup:
        return None
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Clean DOI for filename (replace characters that aren't suitable for filenames)
    clean_doi = doi.replace('/', '_').replace('\\', '_').replace(':', '_')
    filename = f"{clean_doi}.xml"
    filepath = os.path.join(output_dir, filename)
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(xml_soup.prettify())
        print(f"XML saved to {filepath}")
        return filepath
    except Exception as e:
        print(f"Error saving XML for DOI {doi}: {e}")
        return None

def batch_fetch_elsevier_xml(
    doi_list,
    output_dir="elsevier_xml",
    delay=1,
    max_retries=3,
    api_key=None,
    timeout=30,
):
    """
    Fetch XML documents for multiple DOIs with rate limiting.
    
    Parameters:
    -----------
    doi_list : list
        List of DOIs to fetch
    output_dir : str
        Directory where XML files will be saved
    delay : float
        Delay in seconds between API calls to avoid rate limiting
    max_retries : int
        Maximum number of retries for failed requests
        
    Returns:
    --------
    dict
        Dictionary with DOIs as keys and file paths as values
    """
    results = {}
    
    for doi in tqdm(doi_list, desc="Fetching XML documents"):
        # Skip if already downloaded
        clean_doi = doi.replace('/', '_').replace('\\', '_').replace(':', '_')
        filename = f"{clean_doi}.xml"
        filepath = os.path.join(output_dir, filename)
        
        if os.path.exists(filepath):
            print(f"XML for {doi} already exists at {filepath}")
            results[doi] = filepath
            continue
        
        # Try to fetch with retries
        for attempt in range(max_retries + 1):
            xml_soup = fetch_elsevier_xml(doi, api_key=api_key, timeout=timeout)
            
            if xml_soup:
                # Successfully fetched
                file_path = save_xml_to_file(doi, xml_soup, output_dir)
                results[doi] = file_path
                break
            elif attempt < max_retries:
                # Retry after increasing delay
                retry_delay = delay * (2 ** attempt)  # Exponential backoff
                print(f"Retrying DOI {doi} in {retry_delay} seconds (attempt {attempt + 1}/{max_retries})...")
                time.sleep(retry_delay)
            else:
                # All retries failed
                print(f"Failed to fetch XML for DOI {doi} after {max_retries} retries")
                results[doi] = None
        
        # Wait before next request to avoid rate limiting
        time.sleep(delay)
    
    return results

# Example usage
if __name__ == "__main__":
    # Example DOI
    example_doi = "10.1016/j.jclepro.2017.06.175"
    xml_doc = fetch_elsevier_xml(example_doi)
    
    if xml_doc:
        print("Successfully fetched XML document")
        save_xml_to_file(example_doi, xml_doc)
        
        # Example of accessing elements in the XML
        # title = xml_doc.find('dc:title')
        # if title:
        #     print(f"Article title: {title.text.strip()}")
    
    # Example of batch processing
    # doi_list = [
    #     "10.1016/j.jclepro.2017.06.175",
    #     "10.1007/s11367-021-01893-2",
    #     "10.1007/s11367-011-0321-7"
    # ]
    # results = batch_fetch_elsevier_xml(doi_list) 