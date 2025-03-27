from src.nlp.reddit_historical_scraper import EnhancedRedditScraper

# Initialize the scraper
scraper = EnhancedRedditScraper(output_dir="data/processed/test_output")

# Test 1: Test simulated article generation (this doesn't require API credentials)
print("Testing simulated article generation...")
articles = scraper._collect_simulated_articles(
    keywords=scraper.OIL_KEYWORDS[:5],  # Use just a few keywords for testing
    start_date=datetime.datetime(2000, 1, 1),
    end_date=datetime.datetime(2005, 12, 31),
    max_articles=20,  # Generate just a few articles for testing
    source="test-source"
)
print(f"Generated {len(articles)} simulated articles")

# Test 2: Test the financial sentiment data generation
print("\nTesting financial sentiment data generation...")
sentiment_samples = scraper._process_fin_phrasebank_dataset(
    start_year=2013,
    end_year=2014,
    max_samples=15,
    min_relevance=0.3
)
print(f"Generated {len(sentiment_samples)} financial sentiment samples")

# Test 3: Create a small comprehensive dataset with simulated data
print("\nCreating small comprehensive dataset...")
mini_dataset = scraper.create_comprehensive_dataset(
    start_year=2000,
    end_year=2005,
    include_reddit=False,  # Skip Reddit to avoid API dependency
    include_news=True,     # Include simulated news
    include_financial=True # Include simulated financial sentiment
)
print(f"Created dataset with {len(mini_dataset)} items")

# Test 4: Generate visualization for the dataset
if not mini_dataset.empty:
    print("\nGenerating coverage visualization...")
    scraper.visualize_coverage(
        mini_dataset,
        output_file="test_coverage.png"
    )
    print(f"Visualization saved to {scraper.output_dir}/test_coverage.png")

print("\nTests completed!")