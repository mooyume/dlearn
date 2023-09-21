param (
    [Parameter(Mandatory=$true)]
    [string]$FILE
)

$datasets = "ae_photos", "apple2orange", "summer2winter_yosemite", "horse2zebra", "monet2photo", "cezanne2photo", "ukiyoe2photo", "vangogh2photo", "maps", "cityscapes", "facades", "iphone2dslr_flower", "ae_photos"

if ($FILE -notin $datasets) {
    Write-Host "Available datasets are: apple2orange, summer2winter_yosemite, horse2zebra, monet2photo, cezanne2photo, ukiyoe2photo, vangogh2photo, maps, cityscapes, facades, iphone2dslr_flower, ae_photos"
    exit
}

$URL = "https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/$FILE.zip"
$ZIP_FILE = "./$FILE.zip"
$TARGET_DIR = "./$FILE"


# Adapt to project expected directory hierarchy
New-Item -ItemType Directory -Force -Path "$TARGET_DIR/train", "$TARGET_DIR/test"
Move-Item "$TARGET_DIR/trainA" "$TARGET_DIR/train/A"
Move-Item "$TARGET_DIR/trainB" "$TARGET_DIR/train/B"
Move-Item "$TARGET_DIR/testA" "$TARGET_DIR/test/A"
Move-Item "$TARGET_DIR/testB" "$TARGET_DIR/test/B"
