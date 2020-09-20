# NPSound
A collection of simple audio modification methods based on NumPy.

## Supported Formats
- .wav

## Installation
`pip install np-sound`

## Usage
```python
from np_sound import NPSound

sound = NPSound("soundfile.wav")

# Plot soundfile.wav with the title "NPSound Demo"
sound.plot(title="NPSound Demo")

# Reverse the section of audio from 1.5s to 3s
# NPSound objects are immutable, so the 'sound' object will be unchanged
reversed_audio = sound.reverse((1.5, 3))
reversed_audio.plot()

# Amplify the entire audio file by 50% and plot it
sound.amplify(50).plot(title="Amplified by 50%")

# Concatenate 'sound' with its mirrored version and write to a new sound file
mirror = sound + sound.reverse()
mirror.write("mirrored.wav")

# Plot 5 copies of 'sound' side by side
(sound * 5).plot()

# Plot a 50% softened 'sound' on top of the original 'sound'
sound.plot(layered_plots=[sound.amplify(-50)])

# Plot a 'sound' object padded with 5.5s of empty data on either end below the original 'sound'
sound.plot(adjacent_plots=[sound.pad((5.5, 5.5))])

# Trim audio on both ends so that the first and last values are above 100
sound.clip_at_threshold(100)
```