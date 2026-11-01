"""Generate report/report.pdf from report/report.md with embedded images and clean styling."""
import os, sys, re, base64, subprocess

def img_to_base64(path):
    if not os.path.exists(path):
        print(f"Warning: image {path} not found")
        return ""
    ext = os.path.splitext(path)[1].lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{data}"

def markdown_to_html(md_text, base_dir):
    # Convert image references: ![alt](path)
    def repl_img(match):
        alt = match.group(1)
        rel_path = match.group(2)
        full_path = os.path.normpath(os.path.join(base_dir, rel_path))
        b64 = img_to_base64(full_path)
        if b64:
            return f'<div class="fig-box"><img src="{b64}" alt="{alt}" /><div class="fig-caption">{alt}</div></div>'
        return f'<p><em>[Image missing: {alt}]</em></p>'

    html = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', repl_img, md_text)

    # Basic markdown parsing
    lines = html.split('\n')
    out = []
    in_table = False
    table_header = True
    in_list = False

    for line in lines:
        s = line.strip()

        # Handle tables
        if s.startswith('|') and s.endswith('|'):
            if not in_table:
                in_table = True
                table_header = True
                out.append('<table>')
            cells = [c.strip() for c in s[1:-1].split('|')]
            if all(re.match(r'^:?-+:?$', c) for c in cells):
                table_header = False
                continue
            tag = 'th' if table_header else 'td'
            row_html = '<tr>' + ''.join(f'<{tag}>{c}</{tag}>' for c in cells) + '</tr>'
            out.append(row_html)
            continue
        else:
            if in_table:
                out.append('</table>')
                in_table = False

        # Handle lists
        if s.startswith('* ') or s.startswith('- '):
            if not in_list:
                in_list = True
                out.append('<ul>')
            item_text = s[2:]
            out.append(f'<li>{item_text}</li>')
            continue
        elif re.match(r'^\d+\.\s', s):
            if not in_list:
                in_list = True
                out.append('<ol>')
            item_text = re.sub(r'^\d+\.\s', '', s)
            out.append(f'<li>{item_text}</li>')
            continue
        else:
            if in_list:
                out.append('</ul>' if out[-2].startswith('<ul>') or '<ul>' in out[-1] else '</ol>')
                in_list = False

        # Blockquote
        if s.startswith('> '):
            out.append(f'<blockquote>{s[2:]}</blockquote>')
            continue

        # Headers
        if s.startswith('# '):
            out.append(f'<h1>{s[2:]}</h1>')
        elif s.startswith('## '):
            out.append(f'<h2>{s[3:]}</h2>')
        elif s.startswith('### '):
            out.append(f'<h3>{s[4:]}</h3>')
        elif s.startswith('#### '):
            out.append(f'<h4>{s[5:]}</h4>')
        elif s == '---':
            out.append('<hr/>')
        elif s == '':
            out.append('')
        elif not s.startswith('<div class="fig-box">'):
            out.append(f'<p>{s}</p>')
        else:
            out.append(s)

    if in_table:
        out.append('</table>')
    if in_list:
        out.append('</ul>')

    doc = '\n'.join(out)

    # Inline formatting: bold, italic, code, links
    doc = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', doc)
    doc = re.sub(r'\*(.*?)\*', r'<em>\1</em>', doc)
    doc = re.sub(r'`([^`]+)`', r'<code>\1</code>', doc)
    doc = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', doc)

    return doc

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    report_md_path = os.path.join(root, "report", "report.md")
    report_html_path = os.path.join(root, "report", "report.html")
    report_pdf_path = os.path.join(root, "report", "report.pdf")

    with open(report_md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    body_html = markdown_to_html(md_text, os.path.join(root, "report"))

    css = """
    @page {
        size: A4;
        margin: 18mm 16mm 18mm 16mm;
        @bottom-right {
            content: counter(page);
        }
    }
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        font-size: 9.5pt;
        line-height: 1.45;
        color: #24292e;
        background-color: #ffffff;
        margin: 0;
        padding: 0;
    }
    h1 {
        font-size: 18pt;
        font-weight: 700;
        color: #1a202c;
        border-bottom: 2px solid #eaecef;
        padding-bottom: 6px;
        margin-top: 0;
        margin-bottom: 12px;
    }
    h2 {
        font-size: 13pt;
        font-weight: 600;
        color: #2d3748;
        border-bottom: 1px solid #eaecef;
        padding-bottom: 4px;
        margin-top: 18px;
        margin-bottom: 8px;
        page-break-after: avoid;
    }
    h3 {
        font-size: 11pt;
        font-weight: 600;
        color: #4a5568;
        margin-top: 14px;
        margin-bottom: 6px;
        page-break-after: avoid;
    }
    h4 {
        font-size: 10pt;
        font-weight: 600;
        color: #4a5568;
        margin-top: 10px;
        margin-bottom: 4px;
        page-break-after: avoid;
    }
    p {
        margin-top: 0;
        margin-bottom: 8px;
    }
    ul, ol {
        margin-top: 0;
        margin-bottom: 8px;
        padding-left: 20px;
    }
    li {
        margin-bottom: 3px;
    }
    blockquote {
        margin: 10px 0;
        padding: 6px 12px;
        background-color: #f7fafc;
        border-left: 4px solid #3182ce;
        color: #2d3748;
        font-size: 9pt;
        border-radius: 2px;
    }
    code {
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace;
        font-size: 8.5pt;
        background-color: #f0f3f6;
        padding: 2px 4px;
        border-radius: 3px;
    }
    a {
        color: #2b6cb0;
        text-decoration: none;
    }
    hr {
        height: 1px;
        background-color: #e2e8f0;
        border: none;
        margin: 14px 0;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        font-size: 7.8pt;
        margin: 10px 0 14px 0;
        page-break-inside: auto;
    }
    tr {
        page-break-inside: avoid;
        page-break-after: auto;
    }
    th {
        background-color: #f1f5f9;
        color: #1e293b;
        font-weight: 600;
        text-align: left;
        border: 1px solid #cbd5e1;
        padding: 4px 6px;
    }
    td {
        border: 1px solid #e2e8f0;
        padding: 4px 6px;
        vertical-align: top;
    }
    tr:nth-child(even) td {
        background-color: #f8fafc;
    }
    .fig-box {
        text-align: center;
        margin: 12px 0;
        page-break-inside: avoid;
    }
    .fig-box img {
        max-width: 95%;
        height: auto;
        border-radius: 4px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .fig-caption {
        font-size: 8pt;
        color: #718096;
        margin-top: 4px;
        font-style: italic;
    }
    """

    full_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>Computer Vision Intern Task Report - Krishna</title>
<style>
{css}
</style>
</head>
<body>
{body_html}
</body>
</html>
"""

    with open(report_html_path, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"Generated HTML report: {report_html_path} ({os.path.getsize(report_html_path)/1024:.1f} KB)")

    # Render PDF using headless Chrome
    chrome_bin = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    user_data = "/tmp/chrome-user-data"
    os.makedirs(user_data, exist_ok=True)

    cmd = [
        chrome_bin,
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        f"--user-data-dir={user_data}",
        f"--print-to-pdf={report_pdf_path}",
        report_html_path
    ]
    print("Running Chrome headless to generate PDF...")
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print("Chrome error:", res.stderr)
        sys.exit(1)

    if not os.path.exists(report_pdf_path):
        print(f"Error: {report_pdf_path} was not created.")
        sys.exit(1)

    pdf_size_mb = os.path.getsize(report_pdf_path) / (1024 * 1024)
    print(f"Successfully generated {report_pdf_path} ({pdf_size_mb:.2f} MB)")

    # Page count check
    with open(report_pdf_path, "rb") as f:
        pdf_bytes = f.read()
    page_count = len(re.findall(rb"/Type\s*/Page\b", pdf_bytes))
    print(f"PDF Page Count: {page_count} pages")

if __name__ == "__main__":
    main()
