# macOS launchd Setup for Paper-Only Research Sessions

These templates run `research-session` once per schedule tick. Each invocation executes a **bounded paper-only research loop** and exits. They are **not installed automatically**.

Research-only, paper-only; no orders, wallets, or execution endpoints.

caffeinate keeps the Mac awake while the research session process runs. Closing the lid may still sleep some MacBooks unless connected to power/external display depending on macOS settings.

## Files

- Morning plist: `launchd/com.hyperliquid.research.session.morning.plist`
- Evening plist: `launchd/com.hyperliquid.research.session.evening.plist`
- Session reports: `reports/bid_ask_research_scheduler/sessions/`
- Dashboard audit JSON: `reports/bid_ask_research_scheduler/phase3_audit_latest.json`

## Install morning session

```bash
cp launchd/com.hyperliquid.research.session.morning.plist ~/Library/LaunchAgents/
launchctl unload ~/Library/LaunchAgents/com.hyperliquid.research.session.morning.plist 2>/dev/null || true
launchctl load ~/Library/LaunchAgents/com.hyperliquid.research.session.morning.plist
```

## Install evening session

```bash
cp launchd/com.hyperliquid.research.session.evening.plist ~/Library/LaunchAgents/
launchctl unload ~/Library/LaunchAgents/com.hyperliquid.research.session.evening.plist 2>/dev/null || true
launchctl load ~/Library/LaunchAgents/com.hyperliquid.research.session.evening.plist
```

## Check status

```bash
launchctl list | grep hyperliquid.research.session
python3 bid_ask_research_scheduler.py autopilot-status
```

## Uninstall

```bash
launchctl unload ~/Library/LaunchAgents/com.hyperliquid.research.session.morning.plist
launchctl unload ~/Library/LaunchAgents/com.hyperliquid.research.session.evening.plist
rm ~/Library/LaunchAgents/com.hyperliquid.research.session.morning.plist
rm ~/Library/LaunchAgents/com.hyperliquid.research.session.evening.plist
```

Morning schedule: **08:00** local time (`com.hyperliquid.research.session.morning`).
Evening schedule: **16:00** local time (`com.hyperliquid.research.session.evening`).
Session duration: **8** hours; tick interval: **60** minutes.

## Manual caffeinated session (without launchd)

```bash
/usr/bin/caffeinate -d -i -m -s -u /Library/Frameworks/Python.framework/Versions/3.14/bin/python3 /Users/lysto_mac/hyperliquid-bot/bid_ask_research_scheduler.py research-session --duration-hours 8 --tick-interval-minutes 60 --tick-timeout-minutes 90 --requester-pays
```
