"""Main log analyzer module using Ollama for local LLM analysis."""

import requests
import json
from typing import Optional, List
from utils import read_log_file, chunk_logs, extract_error_patterns, summarize_logs
from embedder import LogEmbedder
from prompt import (
    create_analysis_prompt, 
    create_summary_prompt,
    SYSTEM_PROMPT
)


class LogAnalyzer:
    """Main log analyzer using embeddings and local LLM."""
    
    def __init__(self, ollama_host: str = "http://localhost:11434", model: str = "llama3"):
        """
        Initialize the log analyzer.
        
        Args:
            ollama_host: URL to Ollama service
            model: Model name to use (llama3, phi3, etc.)
        """
        self.ollama_host = ollama_host
        self.model = model
        self.embedder = LogEmbedder()
        self.chunks = []
        
    def _is_ollama_available(self) -> bool:
        """Check if Ollama service is available."""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def _call_ollama(self, prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
        """
        Call Ollama API for LLM inference.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt for context
            
        Returns:
            LLM response
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False
        }
        
        try:
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json=payload,
                timeout=300
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                return f"Error: Ollama returned status {response.status_code}"
        except requests.exceptions.ConnectionError:
            return f"Error: Could not connect to Ollama at {self.ollama_host}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def load_logs(self, log_file_path: str, chunk_size: int = 500) -> None:
        """
        Load and chunk log file.
        
        Args:
            log_file_path: Path to log file
            chunk_size: Size of chunks for processing
        """
        print(f"Loading log file: {log_file_path}")
        log_content = read_log_file(log_file_path)
        print(f"✓ Loaded {len(log_content)} characters")
        
        print(f"Chunking logs with size {chunk_size}...")
        self.chunks = chunk_logs(log_content, chunk_size=chunk_size)
        print(f"✓ Created {len(self.chunks)} chunks")
        
        # Build embedding index
        print("Building embedding index...")
        self.embedder.build_index(self.chunks)
    
    def analyze_logs(self, analysis_type: str = "general") -> str:
        """
        Perform comprehensive log analysis.
        
        Args:
            analysis_type: Type of analysis (general, errors, performance, security)
            
        Returns:
            Analysis result from LLM
        """
        if not self.chunks:
            return "Error: No logs loaded. Call load_logs() first."
        
        if not self._is_ollama_available():
            return f"Error: Ollama service not available at {self.ollama_host}. Please start Ollama with: ollama run {self.model}"
        
        print(f"\n🤖 Analyzing logs ({analysis_type} analysis)...")
        
        # Use top chunks for analysis
        top_chunks = self.chunks[:min(5, len(self.chunks))]
        
        prompt = create_analysis_prompt(top_chunks, analysis_type)
        response = self._call_ollama(prompt)
        
        return response
    
    def search_logs(self, query: str, k: int = 5) -> List[tuple]:
        """
        Search logs using semantic similarity.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of (chunk, similarity_score) tuples
        """
        if not self.chunks:
            return []
        
        print(f"\n🔎 Searching for: '{query}'")
        results = self.embedder.search(query, k=k)
        
        for i, (chunk, score) in enumerate(results, 1):
            print(f"\n--- Result {i} (similarity: {score:.2%}) ---")
            print(chunk[:200] + "..." if len(chunk) > 200 else chunk)
        
        return results
    
    def get_quick_summary(self, max_lines: int = 20) -> str:
        """
        Get a quick summary of logs through LLM.
        
        Args:
            max_lines: Max lines to include in summary
            
        Returns:
            Summary from LLM
        """
        if not self.chunks:
            return "No logs loaded."
        
        if not self._is_ollama_available():
            return f"Error: Ollama service not available at {self.ollama_host}"
        
        # Create content summary from all chunks
        summary_text = " ".join(self.chunks)
        
        print("📝 Generating summary...")
        prompt = create_summary_prompt(summary_text)
        response = self._call_ollama(prompt)
        
        return response
    
    def get_error_report(self) -> str:
        """
        Generate an error-focused report.
        
        Returns:
            Error analysis from LLM
        """
        return self.analyze_logs(analysis_type="errors")
    
    def get_performance_report(self) -> str:
        """
        Generate a performance-focused report.
        
        Returns:
            Performance analysis from LLM
        """
        return self.analyze_logs(analysis_type="performance")
    
    def get_security_report(self) -> str:
        """
        Generate a security-focused report.
        
        Returns:
            Security analysis from LLM
        """
        return self.analyze_logs(analysis_type="security")
    
    def get_statistics(self) -> dict:
        """
        Get basic statistics about the logs.
        
        Returns:
            Dictionary with log statistics
        """
        if not self.chunks:
            return {}
        
        full_content = " ".join(self.chunks)
        
        return {
            "total_chunks": len(self.chunks),
            "total_characters": sum(len(chunk) for chunk in self.chunks),
            "average_chunk_size": sum(len(chunk) for chunk in self.chunks) / len(self.chunks),
            "error_patterns": extract_error_patterns(full_content)
        }


def main():
    """Main entry point for the log analyzer."""
    import sys
    
    print("=" * 60)
    print("AI Log Analyzer (Local LLM)")
    print("=" * 60)
    
    # Interactive mode
    log_file = None
    
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    else:
        log_file = input("\n📄 Enter path to log file: ").strip()
    
    if not log_file:
        print("❌ No log file provided.")
        return
    
    # Initialize analyzer
    analyzer = LogAnalyzer(model="phi3")  # You can switch to "llama3" if you have it available
    
    # Check Ollama
    print("\n🔌 Checking Ollama connection...")
    if not analyzer._is_ollama_available():
        print(f"❌ Ollama not available at {analyzer.ollama_host}")
        print(f"   Start Ollama with: ollama run phi3")
        return
    print("✓ Ollama available")
    
    # Load and analyze
    try:
        analyzer.load_logs(log_file)
        
        # Quick statistics
        stats = analyzer.get_statistics()
        print(f"\n📊 Log Statistics:")
        print(f"   Chunks: {stats.get('total_chunks', 0)}")
        print(f"   Total size: {stats.get('total_characters', 0)} characters")
        
        # Get analysis
        print("\n" + "=" * 60)
        print("FULL ANALYSIS")
        print("=" * 60)
        analysis = analyzer.analyze_logs("general")
        print(analysis)
        
        # Interactive search
        print("\n" + "=" * 60)
        print("INTERACTIVE SEARCH")
        print("=" * 60)
        while True:
            query = input("\n🔍 Search logs (or 'quit' to exit): ").strip()
            if query.lower() == 'quit':
                break
            if query:
                analyzer.search_logs(query)
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
