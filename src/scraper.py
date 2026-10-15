import asyncio
import asyncio
from playwright.async_api import async_playwright
import pandas as pd
from datetime import datetime, timedelta
import random
import os

async def scrape_live_prices_stealth(source, destination, date):
    print(f"🕵️‍♂️ Initializing Stealth Automation Engine for {source} ➔ {destination}...")

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--window-size=1920,1080",
            ]
        )

        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            viewport={"width": 1920, "height": 1080},
            locale="en-IN",
            timezone_id="Asia/Kolkata",
            permissions=["geolocation"],
            extra_http_headers={
                "Accept-Language": "en-IN,en;q=0.9",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            }
        )

        page = await context.new_page()

        await page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
            Object.defineProperty(navigator, 'languages', {get: () => ['en-IN', 'en']});
            window.chrome = { runtime: {} };
        """)

        search_url = f"https://www.google.com/search?q=flights+from+{source}+to+{destination}+on+{date}"

        try:
            await page.goto(search_url, wait_until="networkidle", timeout=60000)
            await asyncio.sleep(random.uniform(2.0, 3.5))

            # --- Consent wall handling ---
            for text in ["Accept all", "I agree", "Agree", "Accept"]:
                try:
                    btn = page.locator(f"button:has-text('{text}')")
                    await btn.wait_for(timeout=3000)
                    await btn.click()
                    print(f"✅ Consent wall dismissed ({text})")
                    await asyncio.sleep(2.0)
                    break
                except:
                    continue

            # Save debug screenshot regardless of outcome
            os.makedirs('data', exist_ok=True)
            await page.screenshot(path="data/debug_state.png", full_page=True)
            print("📸 State snapshot saved → data/debug_state.png")

            # --- Wait for flight widget ---
            print("📊 Waiting for flight widget to render...")
            price_found = False

            try:
                await page.wait_for_selector("text=/₹/", timeout=15000)
                price_found = True
                print("✅ Price elements detected in DOM")
            except:
                print("⚠️ No ₹ elements after 15s — checking page state...")

            if not price_found:
                # Scroll to trigger lazy load and retry
                await page.evaluate("window.scrollBy(0, 800)")
                await asyncio.sleep(2.0)
                try:
                    await page.wait_for_selector("text=/₹/", timeout=8000)
                    price_found = True
                    print("✅ Prices appeared after scroll")
                except:
                    print("⚠️ Still no prices — dumping page text for diagnosis")
                    sample = await page.evaluate("() => document.body.innerText.slice(0, 500)")
                    print(f"--- PAGE SAMPLE ---\n{sample}\n---")

            # --- Extract prices ---
            await asyncio.sleep(random.uniform(1.0, 2.0))
            page_text = await page.evaluate("() => document.body.innerText")
            lines = [line.strip() for line in page_text.split('\n') if line.strip()]

            extracted_flights = []
            for i, line in enumerate(lines):
                if "₹" in line or "INR" in line:
                    context_window = lines[max(0, i-2):i+3]
                    extracted_flights.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "route": f"{source}-{destination}",
                        "date": date,
                        "data_chunk": " | ".join(context_window)
                    })
                    if len(extracted_flights) >= 10:
                        break

            if extracted_flights:
                df = pd.DataFrame(extracted_flights)
                output_csv = "data/live_scraped_flights.csv"
                write_header = not os.path.exists(output_csv)
                df.to_csv(output_csv, mode='a', index=False, header=write_header)
                print(f"✅ Logged {len(extracted_flights)} rows → '{output_csv}'")
            else:
                print("❌ No flight prices extracted. Open data/debug_state.png to see what loaded.")

        except Exception as e:
            print(f"❌ Exception: {str(e)}")
            os.makedirs('data', exist_ok=True)
            await page.screenshot(path="data/error_screenshot.png", full_page=True)
            print("📸 Error screenshot → data/error_screenshot.png")

        finally:
            await browser.close()


if __name__ == "__main__":
    target_date = (datetime.now() + timedelta(days=5)).strftime("%Y-%m-%d")
    asyncio.run(scrape_live_prices_stealth("Delhi", "Mumbai", target_date))