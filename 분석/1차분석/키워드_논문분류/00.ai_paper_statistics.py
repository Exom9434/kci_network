import pandas as pd 

def count_unique_papers(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Count the number of unique papers based on the 'title' column
    unique_papers_count = df['제목'].nunique()
    
    return unique_papers_count

def count_unique_papers_by_year(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Group by '발행년도' and count unique papers in each year
    unique_papers_by_year = df.groupby('발행연도')['제목'].nunique()
    
    return unique_papers_by_year

if __name__ == "__main__":
    file_path = 'KCI_AI_논문_기본정보_목록_201601_202512.csv'  # Replace with your actual file path
    unique_papers = count_unique_papers(file_path)
    print(f"Number of unique papers: {unique_papers}")
    unique_papers_by_year = count_unique_papers_by_year(file_path)
    print("Number of unique papers by year:")
    print(unique_papers_by_year)


    