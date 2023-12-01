# Patch Explainations

## SFML OnResize Feature - onresize_win32_2.6.0.patch
Previously there was no way to update while resizing the window. This was because the SFML team didn't handle windows callbacks which
meant that the SFML event handler would have to wait for the onSize win32 event to stop since the win32 is blocking.


I created a custom callback that can be set on any window that allows the user to update the window while updating the size which
would allow for the window to not create a weird blocking because the window itself is unable to actually update until the onsize
event had been completed.


