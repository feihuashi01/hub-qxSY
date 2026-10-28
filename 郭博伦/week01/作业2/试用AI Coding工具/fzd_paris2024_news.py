# -*- coding: utf-8 -*-
"""
樊振东 2024 巴黎奥运会乒乓球男单晋级之路 —— 新闻抓取脚本
================================================================

新闻来源：新华网（www.news.cn，新华社主办，国内最权威的新闻网站之一）
抓取方式：只用 Python 标准库（urllib + re），不需要安装任何第三方库，也无需解析 JS。
    新华网的文章页是服务端渲染的静态 HTML：
      · 标题      -> <title> 标签 / <meta name="description">
      · 正文      -> <div id="detail" class="xl-cont-main"> 里的 <p> 段落
      · 来源      -> <meta name="source">（新华网）
      · 发布时间  -> <meta name="publishdate">
      · URL       -> 手工整理，每场比赛一条报道，与页面内容一一核对过

男单 64 人签表，从 1/32 决赛打到决赛共 6 场，每场比赛取一条新华网报道：
    1/32 决赛  7月29日  4:0 扎姆登科（乌克兰）
    1/16 决赛  7月31日  4:1 黄镇廷（中国香港）
    1/8  决赛  7月31日  4:0 卡纳克·杰哈（美国）
    1/4  决赛  8月 1日  4:3 张本智和（日本）
    半决赛     8月 2日  4:0 菲利克斯·勒布伦（法国）
    决赛       8月 4日  4:1 特鲁斯·莫雷高德（瑞典）

运行：
    python fzd_paris2024_news.py
    python fzd_paris2024_news.py -o D:\\data\\fzd_news.json

输出：JSON 文件（默认 fanzhendong_paris2024_news.json，UTF-8，中文不转义）
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

# ----------------------------------------------------------------------------
# 1. 目标页面的 User-Agent（新华网对不带 UA 的请求会拒绝）
# ----------------------------------------------------------------------------
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9",
}

# ----------------------------------------------------------------------------
# 2. 樊振东巴黎奥运会男单晋级之路：每场比赛一条新华网报道
#    字段：轮次 / 对手 / 比分 / 文章 URL
# ----------------------------------------------------------------------------
NEWS_LIST = [
    {
        "stage": "男单1/32决赛",
        "opponent": "扎姆登科（乌克兰）",
        "score": "4:0",
        "url": "http://www.news.cn/sports/20240729/47eccf44381342d384284afb8754e463/c.html",
    },
    {
        "stage": "男单1/16决赛",
        "opponent": "黄镇廷（中国香港）",
        "score": "4:1",
        "url": "http://www.news.cn/sports/20240731/f9f7c7317c384c4283567c01ebf8e7a0/c.html",
    },
    {
        "stage": "男单1/8决赛",
        "opponent": "卡纳克·杰哈（美国）",
        "score": "4:0",
        "url": "http://www.news.cn/sports/20240801/297746097c4040018efa7da4695b5e87/c.html",
    },
    {
        "stage": "男单1/4决赛",
        "opponent": "张本智和（日本）",
        "score": "4:3",
        "url": "http://www.news.cn/sports/20240802/87b7a37d80a74642a47056d43b486d11/c.html",
    },
    {
        "stage": "男单半决赛",
        "opponent": "菲利克斯·勒布伦（法国）",
        "score": "4:0",
        "url": "http://www.news.cn/sports/20240802/be6bbc8e1c3543cebbd46f085e464efd/c.html",
    },
    {
        "stage": "男单决赛",
        "opponent": "特鲁斯·莫雷高德（瑞典）",
        "score": "4:1",
        "url": "http://www.news.cn/sports/20240805/4a16f21285f6429b812dbb7ac601b992/c.html",
    },
]
# 说明：1/4 决赛另有一篇新华社文字特写，作为“每场一条”之外的补充可自行取用：
#   http://www.news.cn/sports/20240802/9bd30f71d44d4ffc9b1f51d9fdb4a5c8/c.html

DEFAULT_OUTPUT = "fanzhendong_paris2024_news.json"


# ----------------------------------------------------------------------------
# 3. 最简爬虫：下载网页
# ----------------------------------------------------------------------------
def download(url, retries=3, timeout=30):
    """下载网页并返回 HTML 文本，自动尝试 utf-8 / gbk 解码。"""
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
            for encoding in ("utf-8", "gbk"):
                try:
                    return raw.decode(encoding)
                except UnicodeDecodeError:
                    continue
            return raw.decode("utf-8", errors="ignore")
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last_error = error
            if attempt < retries:
                time.sleep(2 * attempt)  # 失败退避后重试
    raise RuntimeError("下载失败：%s（%s）" % (url, last_error))


# ----------------------------------------------------------------------------
# 4. 最简解析：用正则从 HTML 中取出 标题 / 正文 / 来源 / 时间
# ----------------------------------------------------------------------------
def clean_text(text):
    """去掉标签、HTML 实体和首尾空白。"""
    text = re.sub(r"<script.*?</script>", "", text, flags=re.S | re.I)
    text = re.sub(r"<style.*?</style>", "", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", "", text)
    entities = {
        "&emsp;": " ", "&nbsp;": " ", "&quot;": '"', "&amp;": "&",
        "&ldquo;": "“", "&rdquo;": "”", "&mdash;": "—", "&middot;": "·",
        "&#12290;": "。", "&lt;": "<", "&gt;": ">",
    }
    for entity, char in entities.items():
        text = text.replace(entity, char)
    return re.sub(r"[ \t\u3000]+", " ", text).strip()


def extract_meta(html, name):
    """取 <meta name="xxx" content="..."> 的值。"""
    pattern = r'<meta[^>]*name="%s"[^>]*content="([^"]*)"' % re.escape(name)
    match = re.search(pattern, html, flags=re.I)
    if match:
        return clean_text(match.group(1))
    # 兼容 content 写在 name 前面的情况
    pattern = r'<meta[^>]*content="([^"]*)"[^>]*name="%s"' % re.escape(name)
    match = re.search(pattern, html, flags=re.I)
    return clean_text(match.group(1)) if match else ""


def extract_title(html):
    """页面标题：优先 <title>，其次 meta description。"""
    match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.S | re.I)
    title = clean_text(match.group(1)) if match else ""
    if not title:
        title = extract_meta(html, "description")
    # 去掉站点后缀，如 “..._2024巴黎奥运会_新华网”（新华网标题固定用下划线拼接栏目）
    for separator in ("_", "|", "｜"):
        if separator in title:
            title = title.split(separator)[0]
            break
    return title.strip("_- |｜")


def extract_body(html):
    """正文：取 <div id="detail"> 容器内的所有 <p> 段落，拼接为纯文本。"""
    container = re.search(
        r'<div[^>]*id="detail"[^>]*>(.*?)(?:<div[^>]*class="[^"]*(?:footer|editor|related)'
        r'|<!--\s*end\s*-->|</body>)',
        html, flags=re.S | re.I,
    )
    html_slice = container.group(1) if container else html

    paragraphs = []
    for raw in re.findall(r"<p[^>]*>(.*?)</p>", html_slice, flags=re.S | re.I):
        text = clean_text(raw)
        if text and text not in paragraphs:
            paragraphs.append(text)
    return "\n".join(paragraphs)


def extract_source(html):
    """来源：meta source，新华网页面为“新华网”。"""
    source = extract_meta(html, "source")
    if not source:
        match = re.search(r"来源[:：]\s*([^<\s]{2,20})", clean_text(html))
        source = match.group(1) if match else "新华网"
    return source


def extract_publish_time(html):
    """发布时间：meta publishdate，取不到时再从正文里找日期。"""
    publish_time = extract_meta(html, "publishdate")
    if publish_time:
        return publish_time
    match = re.search(r"(\d{4})[-年/](\d{1,2})[-月/](\d{1,2})", clean_text(html))
    if match:
        return "%s-%02d-%02d" % (match.group(1), int(match.group(2)), int(match.group(3)))
    return ""


# ----------------------------------------------------------------------------
# 5. 抓取单条新闻 / 全部新闻
# ----------------------------------------------------------------------------
def parse_news(url, html, extra=None):
    """把一篇文章的 HTML 解析成一条新闻记录。"""
    record = {
        "stage": "",
        "opponent": "",
        "score": "",
        "title": extract_title(html),
        "source": extract_source(html),
        "publish_time": extract_publish_time(html),
        "url": url,
        "content": extract_body(html),
    }
    if extra:
        record.update(extra)
    return record


def crawl_news(news_list=NEWS_LIST, delay=1.0, verbose=True):
    """依次爬取新闻列表，返回记录列表；单条失败不影响其它条目。"""
    records = []
    for index, item in enumerate(news_list, start=1):
        url = item["url"]
        if verbose:
            print("[%d/%d] 正在抓取：%s" % (index, len(news_list), url))
        extra = {k: item[k] for k in ("stage", "opponent", "score") if k in item}
        try:
            html = download(url)
            record = parse_news(url, html, extra)
            record["content_length"] = len(record["content"])
            if verbose:
                print("        标题：%s" % record["title"])
                print("        来源：%s  时间：%s  正文：%d 字"
                      % (record["source"], record["publish_time"], record["content_length"]))
            records.append(record)
        except Exception as error:  # noqa: BLE001 —— 单条失败继续
            if verbose:
                print("        抓取失败：%s" % error)
            record = parse_news(url, "", extra)
            record["error"] = str(error)
            records.append(record)
        if index < len(news_list):
            time.sleep(delay)  # 温和抓取，避免给站点造成压力
    return records


# ----------------------------------------------------------------------------
# 6. 保存为 JSON
# ----------------------------------------------------------------------------
def save_json(records, output_path=DEFAULT_OUTPUT):
    """把抓取结果保存为 JSON 文件（UTF-8，ensure_ascii=False 保留中文）。"""
    data = {
        "topic": "樊振东 2024 巴黎奥运会乒乓球男单晋级之路",
        "source_website": "新华网（www.news.cn）",
        "match_count": len(records),
        "news": records,
    }
    directory = os.path.dirname(os.path.abspath(output_path))
    if directory and not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    return os.path.abspath(output_path)


# ----------------------------------------------------------------------------
# 7. 主流程
# ----------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description="抓取新华网关于樊振东巴黎奥运男单晋级之路的新闻")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT, help="JSON 输出路径")
    parser.add_argument("--delay", type=float, default=1.0, help="每条新闻之间的间隔秒数")
    args = parser.parse_args(argv)

    records = crawl_news(delay=args.delay)
    path = save_json(records, args.output)
    print("\n共抓取 %d 条新闻，已保存到：%s" % (len(records), path))
    return 0


if __name__ == "__main__":
    sys.exit(main())