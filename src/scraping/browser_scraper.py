from parsel import Selector
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from playwright.sync_api._generated import Page

def parse_tweets(selector: Selector):
    """
    returns list of tweets on the page where 1st tweet is the 
    main tweet and the rest are replies. In our case we only want the author's tweets
    """
    results = []
    tweets_consolidated_string=''
    # select all tweets on the page as individual boxes
    # each tweet is stored under <article data-testid="tweet"> box:
    tweets = selector.xpath("//article[@data-testid='tweet']")
    for i, tweet in enumerate(tweets):
        # using data-testid attribute we can get tweet details:
        found = {
            "text": "".join(tweet.xpath(".//*[@data-testid='tweetText']//text()").getall()),
            "username": tweet.xpath(".//*[@data-testid='User-Names']/div[1]//text()").get(),
            "handle": tweet.xpath(".//*[@data-testid='User-Names']/div[2]//text()").get(),
            "datetime": tweet.xpath(".//time/@datetime").get(),
            "verified": bool(tweet.xpath(".//svg[@data-testid='icon-verified']")),
            "url": tweet.xpath(".//time/../@href").get(),
            "image": tweet.xpath(".//*[@data-testid='tweetPhoto']/img/@src").get(),
            "video": tweet.xpath(".//video/@src").get(),
            "video_thumb": tweet.xpath(".//video/@poster").get(),
            "likes": tweet.xpath(".//*[@data-testid='like']//text()").get(),
            "retweets": tweet.xpath(".//*[@data-testid='retweet']//text()").get(),
            "replies": tweet.xpath(".//*[@data-testid='reply']//text()").get(),
            "views": (tweet.xpath(".//*[contains(@aria-label,'Views')]").re("(\d+) Views") or [None])[0],
        }
        # main tweet (not a reply):
        if found["url"]:
            parts = found["url"].split('/')
            username = '/' + parts[1] + '/'
        if i == 0:
            author=username
            found["views"] = tweet.xpath('.//span[contains(text(),"Views")]/../preceding-sibling::div//text()').get()
            found["retweets"] = tweet.xpath('.//a[contains(@href,"retweets")]//text()').get()
            found["quote_tweets"] = tweet.xpath('.//a[contains(@href,"retweets/with_comments")]//text()').get()
            found["likes"] = tweet.xpath('.//a[contains(@href,"likes")]//text()').get()
        else:
            if username!=author:
                continue
        results.append({k: v for k, v in found.items() if v is not None})
        tweets_consolidated_string+=found["text"]+'\n\n'
    
    return tweets_consolidated_string

def simulate_browser(pw, url, wait_for_selector=None):
    browser = pw.chromium.launch(headless=True)
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    context = browser.new_context(user_agent=user_agent, viewport={"width": 1920, "height": 1080})
    # Block images, CSS, JS, videos, audio, and fonts
    context.route('**/*.{png,jpg,jpeg,svg,gif,webp,css,js,mp4,webm,ogg,mp3,wav,flac,woff,woff2,eot,ttf,otf}', lambda route, request: route.abort())
    page = context.new_page()
    try:
        if wait_for_selector:
            page.goto(url)
            page.wait_for_selector(wait_for_selector)
        else:
            page.goto(url, wait_until="networkidle")
    except Exception as e: # Catch all exceptions
        print(f"An error occurred while extracting content from {url}\n{e}\nCould be because the tweet no longer exists")
    finally:
        url = page.url
        content = page.content()
        page.close()
        browser.close()
    return url, content

def scrape_tweet(url: str):
    with sync_playwright() as pw:
        final_url, html = simulate_browser(pw, url, "//article[@data-testid='tweet']")
        selector = Selector(html)
        tweets_string = parse_tweets(selector)
        return final_url, tweets_string
    
def scrape_page(url: str):
    with sync_playwright() as pw:
        final_url, html = simulate_browser(pw, url)
        soup = BeautifulSoup(html, 'html.parser')
        clean_text = soup.get_text()
        return final_url, clean_text