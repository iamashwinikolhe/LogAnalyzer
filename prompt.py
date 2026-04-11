"""Prompt templates for LLM analysis."""


SYSTEM_PROMPT = """You are an expert log analyzer. Your task is to analyze log entries and provide:
1. Summary of main issues found
2. Root cause analysis
3. Recommended actions
4. Risk level assessment (LOW, MEDIUM, HIGH, CRITICAL)

Be concise and actionable in your responses."""


def create_analysis_prompt(log_chunks: list[str], analysis_type: str = "general") -> str:
    """
    Create a prompt for analyzing log chunks.
    
    Args:
        log_chunks: List of relevant log chunks
        analysis_type: Type of analysis (general, errors, performance, security)
        
    Returns:
        Formatted prompt for the LLM
    """
    log_context = "\n\n---\n\n".join(log_chunks)
    
    if analysis_type == "errors":
        return f"""Analyze these log chunks and provide a detailed error analysis:

<LOG_CHUNKS>
{log_context}
</LOG_CHUNKS>

Focus on:
1. All errors and exceptions found
2. Error patterns and sequences
3. Impact on system functionality
4. Recommended fixes

Provide a structured analysis."""
    
    elif analysis_type == "performance":
        return f"""Analyze these log chunks for performance issues:

<LOG_CHUNKS>
{log_context}
</LOG_CHUNKS>

Focus on:
1. Slow operations and timeouts
2. Resource usage patterns
3. Performance bottlenecks
4. Optimization recommendations

Provide a structured analysis."""
    
    elif analysis_type == "security":
        return f"""Analyze these log chunks for security concerns:

<LOG_CHUNKS>
{log_context}
</LOG_CHUNKS>

Focus on:
1. Suspicious activities or patterns
2. Authentication/authorization issues
3. Potential security threats
4. Security recommendations

Provide a structured analysis."""
    
    else:  # general
        return f"""Analyze these log chunks and provide a comprehensive summary:

<LOG_CHUNKS>
{log_context}
</LOG_CHUNKS>

Provide:
1. Executive summary
2. Key issues identified
3. Error/warning counts
4. Recommended actions
5. Risk assessment

Be concise and actionable."""


def create_summary_prompt(log_content: str) -> str:
    """
    Create a prompt for summarizing entire log file.
    
    Args:
        log_content: Full log content
        
    Returns:
        Formatted prompt for the LLM
    """
    return f"""Please provide a brief executive summary of these logs:

<LOG_CONTENT>
{log_content[:5000]}  # Limit to first 5000 chars
</LOG_CONTENT>

Summary should include:
1. Overall health status
2. Main issues (if any)
3. Key metrics/patterns
4. Recommended next steps"""


def create_comparative_prompt(current_logs: str, previous_logs: str) -> str:
    """
    Create a prompt for comparing current and previous logs.
    
    Args:
        current_logs: Current log content
        previous_logs: Previous log content for comparison
        
    Returns:
        Formatted prompt for the LLM
    """
    return f"""Compare these two log sets and highlight what has changed:

<PREVIOUS_LOGS>
{previous_logs[:3000]}
</PREVIOUS_LOGS>

<CURRENT_LOGS>
{current_logs[:3000]}
</CURRENT_LOGS>

Highlight:
1. New errors or issues
2. Issues that have been resolved
3. Changed patterns
4. Improvement or degradation in health"""


def create_anomaly_prompt(log_chunk: str, baseline_description: str = "") -> str:
    """
    Create a prompt for detecting anomalies in logs.
    
    Args:
        log_chunk: Log chunk to analyze
        baseline_description: Description of normal behavior
        
    Returns:
        Formatted prompt for the LLM
    """
    baseline_text = f"\nNormal behavior baseline: {baseline_description}" if baseline_description else ""
    
    return f"""Analyze this log chunk for anomalies and unusual patterns:{baseline_text}

<LOG_CHUNK>
{log_chunk}
</LOG_CHUNK>

Identify:
1. Anomalous behaviors
2. Deviation from normal patterns
3. Severity of anomalies
4. Possible causes"""
