"""
Discord Alert System for DermaOps Pipeline.

This module provides notification functionality to send alerts to Discord
when the ML pipeline completes (success or failure).

Usage:
    from src.utils.alerts import send_discord_alert
    
    # On success
    send_discord_alert("Pipeline completed! F1=0.78", "SUCCESS")
    
    # On failure
    send_discord_alert("Pipeline failed: OOM error", "FAILURE")

Environment Variables:
    DISCORD_WEBHOOK_URL: The Discord webhook URL for notifications.
                        If not set, alerts are logged to console only.
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Literal, Optional

# Try to import requests, fall back to urllib if not available
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    import urllib.request
    import urllib.error
    HAS_REQUESTS = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Alert type definitions
AlertType = Literal["SUCCESS", "FAILURE", "INFO", "WARNING"]

# Discord embed colors (decimal format)
ALERT_COLORS = {
    "SUCCESS": 3066993,   # Green (#2ECC71)
    "FAILURE": 15158332,  # Red (#E74C3C)
    "INFO": 3447003,      # Blue (#3498DB)
    "WARNING": 15105570,  # Orange (#E67E22)
}

# Alert emojis
ALERT_EMOJIS = {
    "SUCCESS": "✅",
    "FAILURE": "❌",
    "INFO": "ℹ️",
    "WARNING": "⚠️",
}


def send_discord_alert(
    message: str,
    alert_type: AlertType = "INFO",
    webhook_url: Optional[str] = None,
    include_timestamp: bool = True,
    extra_fields: Optional[dict] = None
) -> bool:
    """
    Send an alert to Discord via webhook.
    
    If DISCORD_WEBHOOK_URL environment variable is not set and no webhook_url
    is provided, the alert is logged to console only.
    
    Args:
        message: The main alert message.
        alert_type: Type of alert - SUCCESS, FAILURE, INFO, or WARNING.
        webhook_url: Optional webhook URL (overrides env variable).
        include_timestamp: Whether to include timestamp in the embed.
        extra_fields: Optional dict of additional fields to include.
        
    Returns:
        True if alert was sent successfully, False otherwise.
    """
    # Get webhook URL
    url = webhook_url or os.environ.get("DISCORD_WEBHOOK_URL")
    
    # Get timestamp
    timestamp = datetime.now(timezone.utc).isoformat() if include_timestamp else None
    
    # Build the embed
    emoji = ALERT_EMOJIS.get(alert_type, "📢")
    color = ALERT_COLORS.get(alert_type, 3447003)
    
    embed = {
        "title": f"{emoji} DermaOps Pipeline {alert_type}",
        "description": message,
        "color": color,
        "footer": {
            "text": "DermaOps ML Pipeline"
        }
    }
    
    if timestamp:
        embed["timestamp"] = timestamp
    
    # Add extra fields if provided
    if extra_fields:
        embed["fields"] = [
            {"name": k, "value": str(v), "inline": True}
            for k, v in extra_fields.items()
        ]
    
    # Build payload
    payload = {
        "username": "DermaOps Bot",
        "embeds": [embed]
    }
    
    # Log the alert locally regardless
    log_message = f"[{alert_type}] {message}"
    if alert_type == "SUCCESS":
        logger.info(log_message)
    elif alert_type == "FAILURE":
        logger.error(log_message)
    elif alert_type == "WARNING":
        logger.warning(log_message)
    else:
        logger.info(log_message)
    
    # If no webhook URL, just log and return
    if not url:
        logger.info("📢 Discord alert (no webhook configured):")
        logger.info(f"   {emoji} {alert_type}: {message}")
        if extra_fields:
            for k, v in extra_fields.items():
                logger.info(f"   • {k}: {v}")
        return True  # Return True since logging succeeded
    
    # Send to Discord
    try:
        if HAS_REQUESTS:
            response = requests.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            response.raise_for_status()
        else:
            # Fallback to urllib
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status >= 400:
                    raise Exception(f"HTTP {response.status}")
        
        logger.info(f"✅ Discord alert sent successfully: {alert_type}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to send Discord alert: {str(e)}")
        return False


def send_pipeline_success_alert(
    f1_score: float,
    accuracy: float,
    duration_minutes: float,
    webhook_url: Optional[str] = None
) -> bool:
    """
    Send a success alert with pipeline metrics.
    
    Args:
        f1_score: The final F1 score.
        accuracy: The final accuracy.
        duration_minutes: Total pipeline duration in minutes.
        webhook_url: Optional webhook URL.
        
    Returns:
        True if sent successfully.
    """
    message = f"Pipeline completed successfully! 🎉\n\n**Final Metrics:**"
    
    extra_fields = {
        "F1 Score": f"{f1_score:.4f}",
        "Accuracy": f"{accuracy:.4f}",
        "Duration": f"{duration_minutes:.1f} min"
    }
    
    return send_discord_alert(
        message=message,
        alert_type="SUCCESS",
        webhook_url=webhook_url,
        extra_fields=extra_fields
    )


def send_pipeline_failure_alert(
    error_message: str,
    failed_stage: str,
    webhook_url: Optional[str] = None
) -> bool:
    """
    Send a failure alert with error details.
    
    Args:
        error_message: The error message.
        failed_stage: Which stage failed.
        webhook_url: Optional webhook URL.
        
    Returns:
        True if sent successfully.
    """
    message = f"Pipeline failed! 💥\n\n**Error:** {error_message}"
    
    extra_fields = {
        "Failed Stage": failed_stage,
        "Error Type": type(error_message).__name__ if not isinstance(error_message, str) else "Exception"
    }
    
    return send_discord_alert(
        message=message,
        alert_type="FAILURE",
        webhook_url=webhook_url,
        extra_fields=extra_fields
    )


# Convenience functions for quick alerts
def alert_success(message: str) -> bool:
    """Quick success alert."""
    return send_discord_alert(message, "SUCCESS")


def alert_failure(message: str) -> bool:
    """Quick failure alert."""
    return send_discord_alert(message, "FAILURE")


def alert_info(message: str) -> bool:
    """Quick info alert."""
    return send_discord_alert(message, "INFO")


def alert_warning(message: str) -> bool:
    """Quick warning alert."""
    return send_discord_alert(message, "WARNING")


if __name__ == "__main__":
    # Test alerts
    print("Testing Discord Alert System...\n")
    
    # Test each alert type
    send_discord_alert("This is a test SUCCESS message", "SUCCESS")
    send_discord_alert("This is a test FAILURE message", "FAILURE")
    send_discord_alert("This is a test INFO message", "INFO")
    send_discord_alert("This is a test WARNING message", "WARNING")
    
    # Test pipeline-specific alerts
    print("\nTesting pipeline alerts...")
    send_pipeline_success_alert(
        f1_score=0.7784,
        accuracy=0.7635,
        duration_minutes=45.2
    )
    
    send_pipeline_failure_alert(
        error_message="CUDA out of memory",
        failed_stage="Training"
    )
    
    print("\n✅ Alert system test complete!")
