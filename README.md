# Global_Style_Transfer

## run the test code
`bash test.sh`
## Calculate the ArtFID
`cd art-fid`


`CUDA_VISIBLE_DEVICES=0 python3 -m art_fid --style_images path/to/style-images --content_images path/to/content-images --stylized_images path/to/stylized-images`
