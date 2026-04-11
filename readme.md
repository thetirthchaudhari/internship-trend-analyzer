# Internship & Hiring Trend Analyzer

Scrapes LinkedIn job listings, stores them in MongoDB, and runs NLP analysis.

---

## Project Structure

```
internship-trend-analyzer/
├── app.py                          ← Streamlit dashboard (main entry point)
├── scraper/
│   ├── __init__.py
│   └── linkedin_scraper.py         ← Selenium scraper + scrape_and_store()
├── database/
│   ├── __init__.py
│   └── mongo_client.py             ← MongoDB connection, insert, load
├── processing/
│   ├── __init__.py
│   └── data_cleaner.py             ← clean_data() + prepare_for_ml()
├── analysis/
│   ├── __init__.py
│   └── skill_analyzer.py           ← Skill counting + TF-IDF
├── visualization/
│   ├── __init__.py
│   └── chart_generator.py          ← matplotlib charts
└── data/                           ← CSV backups saved here
```

---

## 1. Install MongoDB locally (Mac)

```bash
# Install Homebrew if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Tap the MongoDB formula
brew tap mongodb/brew

# Install MongoDB Community Edition
brew install mongodb-community

# Start MongoDB as a background service
brew services start mongodb-community

# Verify it's running
brew services list | grep mongodb
```

MongoDB will now run at: `mongodb://localhost:27017/`

To stop MongoDB later:
```bash
brew services stop mongodb-community
```

---

## 2. Install Python dependencies

```bash
cd /Users/tirthchaudhari/Desktop/internship-trend-analyzer
source venv/bin/activate
pip install pymongo
```

All other packages should already be installed.

---

## 3. Run the app

```bash
source venv/bin/activate
streamlit run app.py
```

---

## 4. How the pipeline works

```
1. Click "Scrape & Store in MongoDB"
        ↓
2. LinkedInScraper.scrape_jobs() — collects jobs from LinkedIn
        ↓
3. save_to_csv() — saves raw data to data/raw_jobs.csv (backup)
        ↓
4. insert_jobs_bulk() — inserts into MongoDB job_market.internship_jobs
   (duplicates are automatically skipped using MD5 hash of title+company+location)
        ↓
5. clean_data() — lowercase, remove URLs, drop missing fields
        ↓
6. prepare_for_ml() — remove special chars, create 'text' feature column
        ↓
7. get_analysis_summary() — skill counts + TF-IDF keywords
        ↓
8. Charts + Dashboard displayed in Streamlit
```

---

## 5. MongoDB document structure

Database: `job_market`
Collection: `internship_jobs`

```json
{
  "title":       "machine learning intern",
  "company":     "abc tech",
  "location":    "bangalore",
  "description": "looking for python ml sql skills...",
  "source":      "linkedin",
  "scraped_at":  "2026-03-07T12:00:00Z",
  "_id_hash":    "a3f5c2..." 
}
```

The `_id_hash` is an MD5 of `title|company|location` — used to prevent duplicates.

---

## 6. Query MongoDB manually (optional)

```bash
# Open MongoDB shell
mongosh

# Switch to database
use job_market

# Count all stored jobs
db.internship_jobs.countDocuments()

# Find all ML intern jobs
db.internship_jobs.find({ title: /machine learning/ })

# Find jobs by location
db.internship_jobs.find({ location: "bangalore" })

# Delete all documents (reset)
db.internship_jobs.deleteMany({})
```