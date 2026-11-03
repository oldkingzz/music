-- Probe Logic Pro's UI via System Events (run from the Terminal panel, which has Accessibility): bring Logic to the
-- front, list its windows and menu bar items. Read-only apart from activating the app.
tell application "Logic Pro Creator Studio" to activate
delay 1
tell application "System Events"
	tell process "Logic Pro Creator Studio"
		set out to "windows: "
		repeat with w in windows
			set out to out & (name of w as text) & " | "
		end repeat
		set out to out & linefeed & "menus: "
		repeat with m in menu bar items of menu bar 1
			set out to out & (name of m as text) & " | "
		end repeat
	end tell
end tell
return out
