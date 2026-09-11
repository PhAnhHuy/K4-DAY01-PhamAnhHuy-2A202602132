# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:** 11/09/2026

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:** Python

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`): (`468`, `cab`, `1`, `0.510915`, `ImageNet-1K`)
- Record này mô tả toàn ảnh như thế nào? Chủ thể xe 'cab' là chủ thể trung tâm, chiếm nhiều nhất bức ảnh và mô tả toàn bộ bức ảnh
- Ai định nghĩa class list mà checkpoint có thể dự đoán? ImageNet-1K
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy? Có nhiều bộ dữ liệu khác nhau, cùng 1 class_id khác nhau trên các tập như COCO, ImageNet-1K. Cùng một class_name có thể xuất hiện ở nhiều taxonomy nhưng phạm vi hoặc tiêu chuẩn gán nhãn khác nhau.
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì? Guideline cần quy định rõ đây là bài toán Single-label hay Multi-label, quy tắc xác định chủ thể, ngưỡng loại trừ, xử lý nhãn đặc biệt
- Vì sao model score không phải ground truth? Ground truth tượng trưng cho quy tắc chung được con người tạo ra mà mô hình cần hướng tới. Model score là mức độ tự tin của mô hình trên trọng số hiện tại. Model score thường là chỉ là dự đoán của mô hình, dự đoán có thể có model score cao nhưng lại không khớp với ground truth

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):(`person`, `0.769676`, `[
      0.08,
      256.79,
      18.39,
      313.12
    ]`, `18.32`, `56.33`)
- Diễn giải vị trí box bằng lời: 0.08 và 256.79 là vị trị của góc trái trên của box, được chiếu từ góc trái trên của bức ảnh đến góc trái trên của box, 18.39 và 313.12 là vị trí góc phải dưới của box
- So sánh số prediction ở hai threshold: Ngưỡng 0.2 có 17 prediction, 0.35 có 11 prediction, 0.6 có 6 prediction
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem? Khi đô bao phủ tăng, khối lượng các object reviewer cần phải xem sẽ giảm. Số lượng box giảm mạnh giúp màn hình trực quan thoáng hơn, nhưng reviewer phải chuyển sang thao tác chủ động, tự quét mắt tìm vật thể bị sót để vẽ lại từ đầu. Thao tác vẽ một box mới thường tốn thời gian gấp $2 - 3$ lần so với việc bấm chấp nhận một box đã có sẵn.
- Đề xuất một quy tắc box chặt: Quy tắc tiếp xúc cực trị 4 cạnh, Bốn cạnh của bounding box phải tiếp xúc trực tiếp với điểm ảnh ngoài cùng của vật thể theo cả hai trục ngang và dọc. Box phải là hình chữ nhật nhỏ nhất chứa trọn vẹn toàn bộ phần nhìn thấy được của đối tượng
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định? Quy định rõ tỷ lệ phần trăm nhìn thấy được tối thiểu để quyết định gán nhãn hay bỏ qua, Box chỉ ôm khít những điểm ảnh thực sự nhìn thấy được hoặc Annotator phải tưởng tượng và vẽ bao trùm toàn bộ kích thước vật thể.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`): (`traffic-002`, `car`, `0.891092`, 92 điểm, "polygon_xy": [
      [
        497.0,
        287.0
      ],
      [
        497.0,
        287.0
      ],
      [
        492.0,
        287.0
      ],
      [
        491.0,
        287.0
      ],
      [
        488.0,
        284.0
      ],
      [
        488.0,
        283.0
      ],
      [
        487.0,
        282.0
      ],
      [
        487.0,
        281.0
      ],
      [
        486.0,
        280.0
      ],
      [
        486.0,
        279.0
      ],
      [
        483.0,
        276.0
      ],
      [
        483.0,
        275.0
      ],
      [
        482.0,
        274.0
      ],
      [
        482.0,
        273.0
      ],
      [
        481.0,
        272.0
      ],
      [
        481.0,
        271.0
      ],
      [
        480.0,
        270.0
      ],
      [
        480.0,
        269.0
      ],
      [
        479.0,
        268.0
      ],
      [
        479.0,
        265.0
      ],
      [
        434.0,
        265.0
      ],
      [
        431.0,
        268.0
      ],
      [
        431.0,
        269.0
      ])
- Polygon bổ sung chi tiết gì so với box? So với Bounding Box chỉ cung cấp phạm vi bao quát hình chữ nhật, chuỗi tọa độ Polygon bổ sung các tầng thông tin hình học và ngữ nghĩa ở mức pixel
- `instance_id` dùng để làm gì và không phải loại ID nào? `instance_id` dùng để định danh duy nhất một cá thể cụ thể và không đại diện cho loài/chủng loại, như class_id.
- Đề xuất một quy tắc biên mask: Quy tắc chuyển dịch gradient cực đại và ranh giới điểm ảnh 50%, Tại các đường viền chuyển tiếp mờ hoặc bị khử răng cưa (alpha-blended edge), chỉ đưa một pixel vào mask nếu 50% diện tích của pixel đó thuộc về vật thể.
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định? Xác định điểm dừng của mask khi biên bị nhòe vệt dài. Phân định ranh giới giữa lốp xe/chân người với mặt đường. Khi một chiếc xe bị cột đèn hoặc người đi bộ chắn ngang ở giữa chia thành hai nửa tách rời, cần lựa chọn cho phép một instance_id sở hữu cấu trúc Multi-Polygon hoặc tách thành hai instance riêng biệt với nhãn phụ liên kết.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Single-label: class_id hoặc tên lớp chuỗi Multi-label: vector one-hot / binary array ([0, 1, 0, ...]) ứng với taxonomy | Ảnh chứa nhiều chủ thể ngang hàng (vừa có taxi, xe bus, người). Chủ thể quá nhỏ hoặc bị hậu cảnh lấn át. Nhập nhằng giữa bối cảnh tổng thể và vật thể cụ thể. | Áp dụng tie-breaking rules theo guideline. Gán nhãn đơn hoặc đa nhãn đúng chuẩn đầu ra quy định. Flag ảnh mơ hồ lên hàng đợi escalation. | Kiểm tra mức độ đại diện của nhãn đối với toàn ảnh. Đối chiếu nhãn với guideline phân cấp ngữ nghĩa (taxonomy hierarchy). Phát hiện thiên kiến gán nhãn theo cảm tính cá nhân thay vì quy tắc diện tích/độ nét. |
| Phát hiện vật thể | Bounding box hình chữ nhật kèm nhãn lớp | Box quá lỏng (thừa nhiều pixel nền) hoặc cắt cụt chi tiết cực trị của vật thể. Bỏ sót vật thể nhỏ, bị che khuất một phần (occluded), hoặc chạm mép ảnh (truncated). Nhầm lẫn bóng đổ/phản xạ vào kích thước thật của vật thể. | Kéo box theo quy tắc tiếp xúc cực trị 4 cạnh (tight bounding box). Loại bỏ bóng đổ và phản xạ. Xử lý che khuất/cắt mép đúng theo quy chuẩn Modal (chỉ vẽ phần thấy) hay Amodal (vẽ ước lượng) của dự án. | Kiểm tra độ khít (tightness) và độ phủ của 4 cạnh tiếp xúc. Quét sót vật thể nhỏ (Recall check) bằng cách kiểm tra các góc khuất/hậu cảnh. Kiểm tra nhầm lẫn gộp 2 vật thể đứng sát nhau thành một box to duy nhất. |
| Instance segmentation | Mask phân vùng kèm định danh cá thể | Đường biên thô, ăn lấn ra ngoài nền (bleeding) hoặc cắt lẹm vào thân vật thể. Không đục rỗng các vùng thủng/xuyên thấu (nan xe, khoảng hở gầm xe). Nhập nhằng ranh giới tiếp xúc giữa hai vật thể cùng lớp đỗ sát nhau hoặc vùng chuyển tiếp mờ (blur) | Chấm điểm polygon bám sát đường biên gradient tương phản cao nhất. Tạo interior rings để khoét bỏ khoảng trống nền bên trong vật thể. Chia tách chính xác các cá thể tiếp xúc nhau, không gộp mask. | Soi độ lệch biên mask tại mức zoom 100%. Kiểm tra tính toàn vẹn của các hốc rỗng xuyên thấu. Đối chiếu tính đồng nhất giữa instance_id, polygon mask và bounding box ngoại tiếp. |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu: Toàn bộ thông tin định danh cá nhân, bắt buộc phải được làm mờ hoặc che phủ tự động ngay tại khâu tiền xử lý trước khi đẩy lên nền tảng gán nhãn cho annotator hoặc lưu trữ vào kho huấn luyện chung. Dữ liệu ảnh gốc chưa làm mờ phải được cô lập trong môi trường lưu trữ giới hạn.
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho: Lead annotator, AI Engineer, Compliance Officer

## 6. Danh sách bằng chứng

- [X] `classification_predictions.json`
- [X] `detection_predictions.json`
- [X] `segmentation_predictions.json`
- [X] `IMAGE_ATTRIBUTION.md`
- [X] `visuals/classification_top5.png`
- [X] `visuals/detection_predictions.png`
- [X] `visuals/segmentation_prediction.png`
- [X] Ô validation cuối notebook báo `PASS`.
- [X] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
