# Conversation Analysis Tool

A tool for analyzing conversation history by clustering topics and visualizing trends.

## 📝 Featured In

<div align="center">
  <a href="https://muddlemap.substack.com/p/what-i-learned-from-analyzing-2-years" target="_blank">
    <img src="https://img.shields.io/badge/Blog-Read%20the%20Detailed%20Tutorial-blue?style=for-the-badge" alt="Read the Blog" />
  </a>
  
  <a href="https://www.linkedin.com/posts/manojkurien1_what-i-learned-from-analyzing-2-years-of-activity-7313825166885568512-ac85?utm_source=share&utm_medium=member_desktop&rcm=ACoAABtF5ukBcRFWTr7DSFDFQ8nfIva8QIdeB4o" target="_blank">
    <img src="https://img.shields.io/badge/LinkedIn-See%20%20Highlights-0077B5?style=for-the-badge&logo=linkedin" alt="LinkedIn Post" />
  </a>
</div>

## ✨ Features

- Clusters conversations into topics using K-means
- Generates visualizations of conversation trends
- Extracts keywords for automatic topic labeling
- Supports multiple data formats via extensible loader interface

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the tool
python cluster_conversations.py

# Or run with sample data (easier for first-time users)
python cluster_conversations.py --sample
```

The tool will:
1. Process files from the `inputs` directory (or `sample_input` when using --sample)
2. Create output in a timestamped directory (`outputs/run_{timestamp}`) or in the `sample_output` directory (when using --sample)
3. Generate visualizations and data exports

## 📊 Sample Data

A sample dataset is included in `sample_input/sample_conversations.json` for testing or as a template. Use the `--sample` flag to run the tool with this data.

## ⚙️ Configuration

Edit the `Config` class in `cluster_conversations.py` to customize:
- Input/output directories
- Clustering parameters
- Visualization settings

## 📁 Directory Structure

```
project/
├── inputs/                           # Standard input files
├── sample_input/                     # Sample input files for demo
│   └── sample_conversations.json     # Sample template
├── outputs/                          # Generated results
│   └── run_YYYY-MM-DD_HH-MM-SS/      # Timestamped run
├── sample_output/                    # Sample output directory for demo
└── cluster_conversations.py          # Main script
```

## 📋 Output Files

The tool generates several outputs:
- CSV files with clustered data and trends
- Visualizations of topic clusters and trends
- Silhouette score plots for cluster optimization

## 🔌 Adding New Data Formats

Implement the `ConversationLoader` interface to support additional data formats:

```python
class MyFormatLoader(ConversationLoader):
    def load_data(self, files: List[str]) -> List[Dict[str, Any]]:
        # Load raw data from files
        
    def process_data(self, raw_data: List[Dict[str, Any]]) -> pd.DataFrame:
        # Process the raw data into a structured DataFrame
        # Must return a DataFrame with at least these columns:
        # - conversation_id: Unique identifier
        # - text: Text content for analysis
        # - create_time: Unix timestamp
```

Then register your loader in the `get_loader` function.

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.