@ECHO OFF
REM Based on: https://stackoverflow.com/questions/138497/iterate-all-files-in-a-directory-using-a-for-loop
REM and https://www.reddit.com/r/learnpython/comments/2jlv1e/python_script_to_take_an_image_and_cut_it_into_4/
setlocal enabledelayedexpansion
for %%f in (./*.jpg) do (
  echo Processing %%f...
  magick convert %%f -crop 8x6@ +repage +adjoin -resize 500x500 output/%%~nf_%%03d.jpg
)