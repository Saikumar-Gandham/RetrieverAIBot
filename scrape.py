from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import regex as re
import pandas as pd
import time
import logging
from datetime import datetime

# Specify the path to your ChromeDriver
service = Service(executable_path="C:/Users/Surya/RetrieverBot_Capstone/chromedriver_win32/chromedriver.exe")

# Configure logging
logging.basicConfig(
    filename="scraping_log.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
def log_update_date(message):
    """Log a custom message with the current date."""
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"{message} - Date: {current_date}")

# Function to retain only English words and recognized punctuation
def extract_english_text(text):
    pattern = r"[A-Za-z0-9\s.,!?\"'():;-]"
    return ''.join(re.findall(pattern, text))


# Function to check if a page requires login or special access
def requires_login_or_access_denied(driver):
    page_source = driver.page_source.lower()
    login_indicators = [
        "login",
        "access denied",
        "restricted access",
        "please sign in",
        "requires login",
        "permission required",
        "sign in to continue"
    ]
    return any(indicator in page_source for indicator in login_indicators)


# Recursive function to scrape links within the "Courses" section up to a certain depth
def scrape_courses_section(driver, url, section_name, current_depth=1, max_depth=3):
    """Scrape links within a section up to a specified depth."""

    # Print max_depth type to debug
    print(f"Current depth: {current_depth}, Max depth: {max_depth}, Max depth type: {type(max_depth)}")

    if url in visited_links or current_depth > max_depth:
        return  # End recursion if URL has been visited or max depth is reached

    # Mark this link as visited
    visited_links.add(url)

    # Visit the URL
    driver.get(url)
    time.sleep(2)
    wait = WebDriverWait(driver, 10)

    try:
        # Check if the page requires login or special access
        if requires_login_or_access_denied(driver):
            print(f"Restricted access: {url} requires login.")
            results.append({
                'Section': section_name,
                'Link': url,
                'Title': 'Login Required',
                'Text': 'Requires login to access this content.'
            })
            return

        # Extract the 'main-content' div
        main_content_element = wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'main-content')))

        # Extract the title and content
        header_element = main_content_element.find_element(By.CLASS_NAME, 'entry-header')
        title = extract_english_text(header_element.text)

        content_element = main_content_element.find_element(By.CLASS_NAME, 'entry-content')
        content = extract_english_text(content_element.text)

        # Append extracted data to the results list
        results.append({
            'Section': section_name,
            'Link': url,
            'Title': title,
            'Text': content
        })

        # Find all internal links in the 'entry-content' div within 'main-content'
        links_in_content = content_element.find_elements(By.TAG_NAME, 'a')
        internal_links = [link.get_attribute('href') for link in links_in_content if link.get_attribute('href')]

        # Recursively visit each internal link, increasing depth
        for link in internal_links:
            if link and link not in visited_links:
                scrape_courses_section(driver, link, section_name, current_depth + 1, max_depth)

                # Return to the original page after visiting each link
                driver.get(url)
                time.sleep(2)

    except Exception as e:
        print(f"Error scraping {url}: {e}")


def extract_main_content(url, section_name):
    """Extract title and text from the entry-header, and detailed subheading content within entry-content."""
    if url in visited_links:
        return None  # Skip if the URL has already been visited
    visited_links.add(url)

    driver.get(url)
    time.sleep(2)

    try:
        # Wait for the main content to load
        main_content = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'main-content')))

        # Extract title from entry-header
        entry_header = main_content.find_element(By.CLASS_NAME, 'entry-header')
        title = entry_header.find_element(By.CLASS_NAME, 'entry-title').text.strip()

        # Extract general text from entry-content
        entry_content = main_content.find_element(By.CLASS_NAME, 'entry-content')
        general_text = entry_content.get_attribute('innerText').strip()

        # Store the main title and general text for the section
        results.append({
            'Section': 'AI',
            'Link': url,
            'Title': title,
            'Text': general_text
        })

        return main_content

    except Exception as e:
        print(f"Error extracting main content from {url}: {e}")
        return None


def extract_main_content_notitle(url, section_name):
    """Extract title and text from the entry-header, and detailed subheading content within entry-content."""
    if url in visited_links:
        return None  # Skip if the URL has already been visited
    visited_links.add(url)

    driver.get(url)
    time.sleep(2)

    try:
        # Wait for the main content to load
        main_content = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'main-content')))

        # Extract general text from entry-content
        entry_content = main_content.find_element(By.CLASS_NAME, 'entry-content')
        general_text = entry_content.get_attribute('innerText').strip()

        # Store the main title and general text for the section
        results.append({
            'Section': section_name,
            'Link': url,
            'Title': f"{section_name} - Main Content",
            'Text': general_text
        })

        return main_content

    except Exception as e:
        print(f"Error extracting main content from {url}: {e}")
        return None


def extract_collapsible_sections(main_content, url, section_name):
    """Extract collapsible sections and their content within the main-content."""
    sections = main_content.find_elements(By.CLASS_NAME, 'sights-expander-wrapper')
    for section in sections:
        try:
            # Locate the title within the <h5> tag
            title_element = section.find_element(By.TAG_NAME, 'h5')
            title = title_element.text.strip()

            # Expand the section if it's collapsible
            expander_trigger = section.find_element(By.CLASS_NAME, 'sights-expander-trigger')
            expander_trigger.click()
            WebDriverWait(driver, 5).until(
                EC.visibility_of(section.find_element(By.CLASS_NAME, 'sights-expander-content'))
            )

            # Locate the content div that follows
            content_div = section.find_element(By.CLASS_NAME, 'sights-expander-content')
            text = content_div.text.strip()

            # Store the result with section name
            results.append({
                'Section': section_name,
                'Link': url,
                'Title': title,
                'Text': text
            })

        except Exception as e:
            print(f"Error extracting section from {url}: {e}")


def scrape_internal_links(main_content, section_name):
    """Identify and visit internal links from the main content."""
    try:
        # Find all anchor tags within the entry-content div
        anchor_tags = main_content.find_elements(By.TAG_NAME, 'a')
        links_to_visit = []

        for anchor in anchor_tags:
            href = anchor.get_attribute('href')
            if href and "dil.umbc.edu" in href and href not in visited_links:
                links_to_visit.append(href)

        # Visit each link and extract data
        for link in links_to_visit:
            main_content = extract_main_content(link, section_name)
            if main_content:
                extract_collapsible_sections(main_content, link, section_name)

    except Exception as e:
        print(f"Error finding internal links: {e}")


def scrape_sidebar_links(url, section_name):
    """Scrape all links found in the sidebar and return to the main page after each."""
    driver.get(url)
    time.sleep(2)

    try:
        sidebar_links = driver.find_elements(By.CSS_SELECTOR, '.sidebar a')
        for link in sidebar_links:
            href = link.get_attribute('href')
            if href and "dil.umbc.edu" in href and href not in visited_links:
                # Scrape content from each sidebar link
                extract_main_content(href, section_name)
                time.sleep(2)

                # Return to the main page
                driver.get(url)
                time.sleep(2)

    except Exception as e:
        print(f"Error extracting sidebar links from {url}: {e}")


def extract_dropdown_content(url, section_name):
    """Extract content from each dropdown section on the page."""
    driver.get(url)
    time.sleep(2)

    try:
        # Locate all dropdown sections on the page
        dropdown_sections = driver.find_elements(By.CLASS_NAME, 'sights-expander-wrapper')
        if not dropdown_sections:
            print(f"No dropdown sections found for {section_name}")

        for section in dropdown_sections:
            try:
                # Locate the question element (title) within the dropdown trigger
                question_element = section.find_element(By.CLASS_NAME, 'mceEditable')
                question_text = question_element.text.strip()

                # Expand the dropdown by clicking the trigger element
                expander_trigger = section.find_element(By.CLASS_NAME, 'sights-expander-trigger')
                driver.execute_script("arguments[0].click();", expander_trigger)
                time.sleep(1)  # Allow time for expansion

                # Wait for content to load in the dropdown
                answer_id = expander_trigger.get_attribute('aria-controls')
                answer_content = WebDriverWait(driver, 5).until(
                    EC.visibility_of_element_located((By.ID, answer_id))
                )

                # Extract the answer text
                answer_text = answer_content.find_element(By.CLASS_NAME, 'mceEditable').get_attribute(
                    'innerText').strip()

                # Store each FAQ entry in results
                results.append({
                    'Section': section_name,
                    'Link': url,
                    'Title': question_text,
                    'Text': answer_text
                })
                print(f"Dropdown content '{question_text}' extracted.")

            except Exception as e:
                print(f"Error extracting dropdown content in {url}: {e}")

    except Exception as e:
        print(f"Error loading dropdown sections from {url}: {e}")


# Set up Chrome options
chrome_options = Options()
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
# chrome_options.add_argument("--headless")  # Uncomment to run in headless mode

# Initialize the WebDriver
driver = webdriver.Chrome(service=service, options=chrome_options)

# List of main sections with URLs
sections = {
    "Courses": 'https://dil.umbc.edu/courses/',
    "Pathways & Certificates": 'https://dil.umbc.edu/pathways-and-certificates/',
    "Advising & Resources": 'https://dil.umbc.edu/resources/',
    "Policies": 'https://dil.umbc.edu/policies/',
    "Prospective Students": 'https://dil.umbc.edu/prospective-students/',
    "Faculty": 'https://dil.umbc.edu/faculty/'
}

# Initialize a list to store the results
results = []
visited_links = set()

# Main scraping process for each section
try:
    logging.info("Scraping started for dil.umbc.edu")
    for section_name, url in sections.items():
        print(f"Scraping section: {section_name}")

        if section_name == "Courses":
            scrape_courses_section(driver, url, section_name)

        # Step 1: Extract content from the main page
        main_content = extract_main_content(url, section_name)

        # Step 2: Extract collapsible sections (if any)
        if main_content:
            extract_collapsible_sections(main_content, url, section_name)

            # Step 3: Identify and visit internal links within the main content
            scrape_internal_links(main_content, section_name)

            # Step 4: Scrape all links in the sidebar
            scrape_sidebar_links(url, section_name)

finally:
    # Log the end of the scraping process
    logging.info("Scraping completed for dil.umbc.edu")
    # Close the browser
    driver.quit()

# Create a DataFrame from the collected data and reorder columns
df_dil = pd.DataFrame(results, columns=['Section', 'Link', 'Title', 'Text'])

# Save the DataFrame to an CSV file
df_dil.to_csv('Data/dil_scraped_data.csv', index=False)

# Set up Chrome options
chrome_options = Options()
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
# chrome_options.add_argument("--headless")  # Uncomment to run in headless mode

# Initialize the WebDriver
driver = webdriver.Chrome(service=service, options=chrome_options)

# Initialize a list to store the results
results = []
visited_links = set()

# Define the F-1 Students sub-sections with their URLs
sub_sections = {
    "Current Students: General": "https://isss.umbc.edu/international-students-f-1/current-students/",
    "Current Students: Employment": "https://isss.umbc.edu/f-1-students/current-students-employment/",
    "Working On-Campus": "https://isss.umbc.edu/international-students-f-1/current-students-employment/working-on-campus/",
    "Economoic Hardship Work Authorization": "https://isss.umbc.edu/international-students-f-1/current-students-employment/economic-hardship-work-authorization/",
    "Understanding Your Documents": "https://isss.umbc.edu/f-1-students/understanding-your-documents/",
    "Social Security Number (SSN)": "https://isss.umbc.edu/f-1-students/social-security-number/",
    "US Taxes": "https://isss.umbc.edu/international-students-f-1/current-students/understanding-your-tax-documents/",
    "Applying for a Maryland Driver’s License & Getting a State ID": "https://isss.umbc.edu/resources/transportation/driving-in-maryland-and-getting-a-state-id/",
    "Change of Immigration Status": "https://isss.umbc.edu/change-of-status/"
}

# List of URLs with dropdowns
dropdown_pages = {
    "Internships and International Students": "https://dil.umbc.edu/resources/internships-and-international-students/",
    "OPT and OPT STEM Information": "https://isss.umbc.edu/opt-and-opt-stem-information/",
    "Working Off-Campus": "https://isss.umbc.edu/international-students-f-1/current-students-employment/working-off-campus/"
}

try:
    logging.info("Scraping started for ISSS website")
    # Scrape each specified sub-section and its sidebar links
    for sub_section_name, link in sub_sections.items():
        print(f"Scraping section: {sub_section_name}")

        # Step 1: Extract main content
        main_content = extract_main_content_notitle(link, sub_section_name)

    # Scrape each page with dropdowns
    for section_name, url in dropdown_pages.items():
        print(f"Scraping dropdown section: {section_name}")

        # Step 1: Extract content from the main page
        main_content = extract_main_content_notitle(url, section_name)

        # Step 2: Extract collapsible sections (if any)
        if main_content:
            extract_dropdown_content(url, section_name)

finally:
    logging.info("Scraping completed for ISSS website")
    driver.quit()

df_isss = pd.DataFrame(results, columns=['Section', 'Link', 'Title', 'Text'])

# Save the DataFrame to an CSV file
df_isss.to_csv('Data/isss_scraped_data.csv', index=False)

# Set up Chrome options
chrome_options = Options()
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
# chrome_options.add_argument("--headless")  # Uncomment to run in headless mode

# Define research links with proper formatting
Research_links = {
    "CSEE Research Areas": "https://www.csee.umbc.edu/csee-research-areas/",
    "Research Centers": "https://www.csee.umbc.edu/research-focus-areas-and-centers/",
    "Research Labs CSEE": "https://www.csee.umbc.edu/research/research-labs/",
    "AI Home": "https://ai.umbc.edu/",
    "Research faculty": "https://ai.umbc.edu/ai-faculty/",
    "Research Labs AI": "https://ai.umbc.edu/labs-groups/",
    "Cyber Security labs": "https://cybersecurity.umbc.edu/training/labs/"
}

driver = webdriver.Chrome(service=service, options=chrome_options)

# Initialize a list to store the results
results = []
visited_links = set()

# Define the main scraping process
try:
    logging.info("Scraping started for research data")
    for section_name, url in Research_links.items():
        print(f"Scraping section: {section_name}")
        # Extract content from the main page for other sections
        main_content = extract_main_content(url, section_name)
        # Step 2: Extract collapsible sections (if any)
        if main_content:
            extract_collapsible_sections(main_content, url, section_name)

            # Step 3: Identify and visit internal links within the main content
            scrape_internal_links(main_content, section_name)

            # Step 4: Scrape all links in the sidebar
            scrape_sidebar_links(url, section_name)

finally:
    logging.info("Scraping completed for research data")
    driver.quit()

# Convert results to a DataFrame
df_csee = pd.DataFrame(results, columns=['Section', 'Link', 'Title', 'Text'])
# Save the DataFrame to an CSV file
df_csee.to_csv('Data/research_data.csv', index=False)

import subprocess

def git_operations():
            try:
                # Stage the CSV files
                subprocess.run([
                    "git", "add",
                    "dil_scraped_data.csv",
                    "isss_scraped_data.csv",
                    "research_data.csv",
                ], check=True)

                # Force add scraping_log.log
                subprocess.run(["git", "add", "-f", "scraping_log.log"], check=True)

                # Commit the changes
                subprocess.run(["git", "commit", "-m", "Update scraped data files"], check=True)

                # Push to the remote repository
                subprocess.run(["git", "push"], check=True)

                print("CSV files pushed to Git successfully.")
            except subprocess.CalledProcessError as e:
                print(f"Error during Git push: {e}")


if __name__ == "__main__":
    git_operations()
