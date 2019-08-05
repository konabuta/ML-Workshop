# Dynamic Time Warping による時系列クラスタリング

時間がずれている時系列データの類似度を周期や動きを基準に判断する手法

<img src="../../docs/dtw-image.png" width=1000> 

Workshop では階層型クラスタリングが実装されているという理由から dtaidistance を利用していますが、他にも Dynamic Time Warping が実装されているライブラリは多数あります。(下記参照)
## Dynamic Time Warping ライブラリ
- Python Library
    - [dtaidistance](https://pypi.org/project/dtaidistance/#description)
    - [fastdtw](https://github.com/slaypni/fastdtw)

    - [dtw](https://github.com/pierre-rouanet/dtw)
    - [tslearn](https://github.com/rtavenar/tslearn)
    - [ucrdtw](http://www.cs.ucr.edu/~eamonn/SIGKDD_trillion.pdf) 
- R Library
    - dtw