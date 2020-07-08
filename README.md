# AIR-Clothing-MA
This is a part of ETRI AIR project. The AIR-Clothing-MA(Multi Attributes) is a kind of mutli-attributes classifier for clothings and their multi attributes.

## Dependencies
-   python >= 3.6 
-   pytorch >= 1.2

-   RoiAlign 
    1. Download and install RoiAlign module for pytorch from [here](https://github.com/longcw/RoIAlign.pytorch)

## Installation
Recommend this procedure!!  (from ubuntu 18.04)
1. download & install cuda 10.2 toolkit 
 - [here](https://developer.nvidia.com/cuda-10.2-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=debnetwork)
2. download & install anaconda python 3.7 version (Linux x86) 
 - [download](https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh)
 - $ sh Anaconda3-2020.02-Linux-x86_64.sh
3. install pytorch
 - (base)$ conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
4. install opencv, matplotlib
 - (base)$ pip install opencv-python, matplotlib
5. install RoIAlign
 - [RoIAlign](https://github.com/longcw/RoIAlign.pytorch)
 
## How to test the model

1.   Download yolo-v3 model from [here](https://drive.google.com/file/d/1yCz6pc6qHJD2Zcz8ldDmJ3NzE8wjaiT6/view?usp=sharing) and put in 'Air-Clothing-MA root directory'.  
2.   Downoad Clothing-MA model from [here](https://drive.google.com/file/d/1k3lvA96ZstbV4a_QtYTuohY79xg_nJYe/view?usp=sharing) and put in 'Air-Clothing-MA root directory'.
3.   Run 'file_demo.py' to run each image file demostration
4.   Run 'cam_demo.py' to run web-cam demostration

## Clothing multi-attributes definition

<html><head><meta content="text/html; charset=UTF-8" http-equiv="content-type"><style type="text/css">ol{margin:0;padding:0}table td,table th{padding:0}.c1{border-right-style:solid;padding:2pt 2pt 2pt 2pt;border-bottom-color:#000000;border-top-width:1pt;border-right-width:1pt;border-left-color:#000000;vertical-align:bottom;border-right-color:#000000;border-left-width:1pt;border-top-style:solid;border-left-style:solid;border-bottom-width:1pt;width:37.7pt;border-top-color:#000000;border-bottom-style:solid}.c7{border-right-style:solid;padding:2pt 2pt 2pt 2pt;border-bottom-color:#000000;border-top-width:1pt;border-right-width:1pt;border-left-color:#000000;vertical-align:bottom;border-right-color:#000000;border-left-width:1pt;border-top-style:solid;border-left-style:solid;border-bottom-width:1pt;width:43.3pt;border-top-color:#000000;border-bottom-style:solid}.c2{border-right-style:solid;padding:2pt 2pt 2pt 2pt;border-bottom-color:#000000;border-top-width:1pt;border-right-width:1pt;border-left-color:#000000;vertical-align:bottom;border-right-color:#000000;border-left-width:1pt;border-top-style:solid;border-left-style:solid;border-bottom-width:1pt;width:31.1pt;border-top-color:#000000;border-bottom-style:solid}.c5{border-right-style:solid;padding:2pt 2pt 2pt 2pt;border-bottom-color:#000000;border-top-width:1pt;border-right-width:1pt;border-left-color:#000000;vertical-align:bottom;border-right-color:#000000;border-left-width:1pt;border-top-style:solid;border-left-style:solid;border-bottom-width:1pt;width:33.6pt;border-top-color:#000000;border-bottom-style:solid}.c3{border-right-style:solid;padding:2pt 2pt 2pt 2pt;border-bottom-color:#000000;border-top-width:1pt;border-right-width:1pt;border-left-color:#000000;vertical-align:bottom;border-right-color:#000000;border-left-width:1pt;border-top-style:solid;border-left-style:solid;border-bottom-width:1pt;width:35.7pt;border-top-color:#000000;border-bottom-style:solid}.c0{color:#000000;font-weight:400;text-decoration:none;vertical-align:baseline;font-size:10pt;font-family:"Arial";font-style:normal}.c9{color:#000000;font-weight:400;text-decoration:none;vertical-align:baseline;font-size:11pt;font-family:"Arial";font-style:normal}.c4{padding-top:0pt;padding-bottom:0pt;line-height:1.15;text-align:left;height:11pt}.c11{border-spacing:0;border-collapse:collapse;margin-right:auto}.c6{padding-top:0pt;padding-bottom:0pt;line-height:1.15;text-align:center}.c13{background-color:#ffffff;max-width:468pt;padding:72pt 72pt 72pt 72pt}.c12{orphans:2;widows:2}.c10{height:27pt}.c8{height:15.8pt}.title{padding-top:0pt;color:#000000;font-size:26pt;padding-bottom:3pt;font-family:"Arial";line-height:1.15;page-break-after:avoid;orphans:2;widows:2;text-align:left}.subtitle{padding-top:0pt;color:#666666;font-size:15pt;padding-bottom:16pt;font-family:"Arial";line-height:1.15;page-break-after:avoid;orphans:2;widows:2;text-align:left}li{color:#000000;font-size:11pt;font-family:"Arial"}p{margin:0;color:#000000;font-size:11pt;font-family:"Arial"}h1{padding-top:20pt;color:#000000;font-size:20pt;padding-bottom:6pt;font-family:"Arial";line-height:1.15;page-break-after:avoid;orphans:2;widows:2;text-align:left}h2{padding-top:18pt;color:#000000;font-size:16pt;padding-bottom:6pt;font-family:"Arial";line-height:1.15;page-break-after:avoid;orphans:2;widows:2;text-align:left}h3{padding-top:16pt;color:#434343;font-size:14pt;padding-bottom:4pt;font-family:"Arial";line-height:1.15;page-break-after:avoid;orphans:2;widows:2;text-align:left}h4{padding-top:14pt;color:#666666;font-size:12pt;padding-bottom:4pt;font-family:"Arial";line-height:1.15;page-break-after:avoid;orphans:2;widows:2;text-align:left}h5{padding-top:12pt;color:#666666;font-size:11pt;padding-bottom:4pt;font-family:"Arial";line-height:1.15;page-break-after:avoid;orphans:2;widows:2;text-align:left}h6{padding-top:12pt;color:#666666;font-size:11pt;padding-bottom:4pt;font-family:"Arial";line-height:1.15;page-break-after:avoid;font-style:italic;orphans:2;widows:2;text-align:left}</style></head><body class="c13"><p class="c4 c12"><span class="c9"></span></p><a id="t.a6c6345e9a73b3ad660c800fb95cf5cfd611401e"></a><a id="t.0"></a><table class="c11"><tbody><tr class="c8"><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">1</span></p></td><td class="c1" colspan="1" rowspan="1"><p class="c6"><span class="c0">2</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">3</span></p></td><td class="c5" colspan="1" rowspan="1"><p class="c6"><span class="c0">4</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">5</span></p></td><td class="c7" colspan="1" rowspan="1"><p class="c6"><span class="c0">6</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">7</span></p></td><td class="c3" colspan="1" rowspan="1"><p class="c6"><span class="c0">8</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">9</span></p></td><td class="c1" colspan="1" rowspan="1"><p class="c6"><span class="c0">10</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">11</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">12</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">13</span></p></td></tr><tr class="c10"><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">GT values</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">Top color(14)</span></p></td><td class="c1" colspan="1" rowspan="1"><p class="c6"><span class="c0">Top pattern(6)</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">Top sex(2)</span></p></td><td class="c5" colspan="1" rowspan="1"><p class="c6"><span class="c0">Top season(4)</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">Top type(7)</span></p></td><td class="c7" colspan="1" rowspan="1"><p class="c6"><span class="c0">Top sleeves(3)</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">Bottom color(14)</span></p></td><td class="c3" colspan="1" rowspan="1"><p class="c6"><span class="c0">Bottom pattern(6)</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">Bottom sex(2)</span></p></td><td class="c1" colspan="1" rowspan="1"><p class="c6"><span class="c0">Bottom season(4)</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">Bottom length(2)</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">Bottom type(2)</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">leg pose(3)</span></p></td></tr><tr class="c8"><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">0</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">null</span></p></td><td class="c1" colspan="1" rowspan="1"><p class="c6"><span class="c0">null</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">null</span></p></td><td class="c5" colspan="1" rowspan="1"><p class="c6"><span class="c0">null</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">null</span></p></td><td class="c7" colspan="1" rowspan="1"><p class="c6"><span class="c0">null</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">null</span></p></td><td class="c3" colspan="1" rowspan="1"><p class="c6"><span class="c0">null</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">null</span></p></td><td class="c1" colspan="1" rowspan="1"><p class="c6"><span class="c0">null</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">null</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">null</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">null</span></p></td></tr><tr class="c10"><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">1</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">white</span></p></td><td class="c1" colspan="1" rowspan="1"><p class="c6"><span class="c0">plain</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">man</span></p></td><td class="c5" colspan="1" rowspan="1"><p class="c6"><span class="c0">spring</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">shirt</span></p></td><td class="c7" colspan="1" rowspan="1"><p class="c6"><span class="c0">short sleeves</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">white</span></p></td><td class="c3" colspan="1" rowspan="1"><p class="c6"><span class="c0">plain</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">man</span></p></td><td class="c1" colspan="1" rowspan="1"><p class="c6"><span class="c0">spring</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">short pants</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">pants</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">standing</span></p></td></tr><tr class="c10"><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">2</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">black</span></p></td><td class="c1" colspan="1" rowspan="1"><p class="c6"><span class="c0">checker</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">woman</span></p></td><td class="c5" colspan="1" rowspan="1"><p class="c6"><span class="c0">summer</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">jumper</span></p></td><td class="c7" colspan="1" rowspan="1"><p class="c6"><span class="c0">long sleeves</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">black</span></p></td><td class="c3" colspan="1" rowspan="1"><p class="c6"><span class="c0">checker</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">woman</span></p></td><td class="c1" colspan="1" rowspan="1"><p class="c6"><span class="c0">summer</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">long pants</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">skirt</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">sitting</span></p></td></tr><tr class="c8"><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">3</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">gray</span></p></td><td class="c1" colspan="1" rowspan="1"><p class="c6"><span class="c0">dotted</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c5" colspan="1" rowspan="1"><p class="c6"><span class="c0">autunm</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">jacket</span></p></td><td class="c7" colspan="1" rowspan="1"><p class="c6"><span class="c0">no sleeves</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">gray</span></p></td><td class="c3" colspan="1" rowspan="1"><p class="c6"><span class="c0">dotted</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c1" colspan="1" rowspan="1"><p class="c6"><span class="c0">autunm</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">lying</span></p></td></tr><tr class="c8"><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">4</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">pink</span></p></td><td class="c1" colspan="1" rowspan="1"><p class="c6"><span class="c0">floral</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c5" colspan="1" rowspan="1"><p class="c6"><span class="c0">winter</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">vest</span></p></td><td class="c7" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">pink</span></p></td><td class="c3" colspan="1" rowspan="1"><p class="c6"><span class="c0">floral</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c1" colspan="1" rowspan="1"><p class="c6"><span class="c0">winter</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td></tr><tr class="c8"><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">5</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">red</span></p></td><td class="c1" colspan="1" rowspan="1"><p class="c6"><span class="c0">striped</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c5" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">parka</span></p></td><td class="c7" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">red</span></p></td><td class="c3" colspan="1" rowspan="1"><p class="c6"><span class="c0">striped</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c1" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td></tr><tr class="c8"><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">6</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">green</span></p></td><td class="c1" colspan="1" rowspan="1"><p class="c6"><span class="c0">mixed</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c5" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">coat</span></p></td><td class="c7" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">green</span></p></td><td class="c3" colspan="1" rowspan="1"><p class="c6"><span class="c0">mixed</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c1" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td></tr><tr class="c8"><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">7</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">blue</span></p></td><td class="c1" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c5" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">dress</span></p></td><td class="c7" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">blue</span></p></td><td class="c3" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c1" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td></tr><tr class="c8"><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">8</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">brown</span></p></td><td class="c1" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c5" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c7" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">brown</span></p></td><td class="c3" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c1" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td></tr><tr class="c8"><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">9</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">navy</span></p></td><td class="c1" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c5" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c7" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">navy</span></p></td><td class="c3" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c1" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td></tr><tr class="c8"><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">10</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">beige</span></p></td><td class="c1" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c5" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c7" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">beige</span></p></td><td class="c3" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c1" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td></tr><tr class="c8"><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">11</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">yellow</span></p></td><td class="c1" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c5" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c7" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">yellow</span></p></td><td class="c3" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c1" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td></tr><tr class="c8"><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">12</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">purple</span></p></td><td class="c1" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c5" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c7" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">purple</span></p></td><td class="c3" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c1" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td></tr><tr class="c8"><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">13</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">orange</span></p></td><td class="c1" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c5" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c7" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">orange</span></p></td><td class="c3" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c1" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td></tr><tr class="c8"><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">14</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">mixed</span></p></td><td class="c1" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c5" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c7" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c6"><span class="c0">mixed</span></p></td><td class="c3" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c1" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c4"><span class="c0"></span></p></td></tr></tbody></table><p class="c4 c12"><span class="c9"></span></p></body></html>			

## LICENSE
This software is a part of AIR, and follows the [AIR License and Service Agreement](LICENSE.md).

## Acknowledgement
* This work was supported by the ICT R&D program of MSIP/IITP. [2017-0-00162, Development of Human-care Robot Technology for Aging Society]   
