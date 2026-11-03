-- v2 of logic_probe_v1: per Logic window, its name, position, size and the names / descriptions of its buttons.
tell application "System Events"
	tell process "Logic Pro Creator Studio"
		set out to ""
		repeat with w in windows
			set out to out & "[" & (name of w as text) & "] pos " & (position of w as text) & " size " & (size of w as text) & linefeed
			repeat with b in (every button of w)
				set out to out & "   button: " & (name of b as text) & " / " & (description of b as text) & linefeed
			end repeat
		end repeat
	end tell
end tell
return out
