import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image

# --- 頁面設定 ---
st.set_page_config(page_title="HK GBP Auto-Checker", layout="wide")

st.title("🏗️ 香港建築圖則自動檢查系統 (GBP Auto-Checker)")
st.markdown("**針對香港 GBP / A&A 圖則設計的自動面積計算 MVP**")

# --- 側邊欄：控制台 ---
st.sidebar.header("1. 圖則輸入")
uploaded_file = st.sidebar.file_uploader("上傳平面圖 (JPG/PNG)", type=["jpg", "png", "jpeg"])

st.sidebar.header("2. 參數設定")
# 這是給測量師調整的「魔術棒靈敏度」
threshold_val = st.sidebar.slider("二值化閾值 (Threshold)", 0, 255, 180, help="越低越黑，越高越白。用來過濾淡色的雜訊。")
blur_kernel = st.sidebar.slider("雜訊模糊 (Blur)", 1, 15, 7, step=2, help="數值越大，越能把斷開的牆連起來，但也可能糊掉窗戶。")
min_area_filter = st.sidebar.number_input("最小房間面積過濾 (m²)", value=2.0, help="小於這個面積的區域會被視為雜訊（如柱子、廁所管道）。")

st.sidebar.header("3. 比例尺校正")
scale_ratio = st.sidebar.number_input("比例尺 (像素/米)", value=50.0, help="例如：圖上 50px 代表現實 1米")

# --- 核心處理函數 (包含高亮邏輯) ---
def process_image(pil_image, thresh_val, blur_val, scale, min_area_m2, highlight_id=None):
    # PIL 轉 OpenCV 格式
    img_cv = np.array(pil_image.convert('RGB'))
    img_cv = img_cv[:, :, ::-1].copy() # RGB to BGR
    original = img_cv.copy()
    
    # 獲取圖片尺寸，用於計算字體大小 (動態調整)
    h, w = original.shape[:2]
    # 經驗公式：圖片越寬，字越大
    dynamic_font_scale = max(0.5, w / 1500.0)
    dynamic_thickness = max(1, int(w / 600.0))

    # 1. 轉灰階
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # 2. 圖像預處理 (Blur + Threshold)
    blurred = cv2.GaussianBlur(gray, (blur_val, blur_val), 0)
    _, thresh = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)
    
    # 3. 形態學操作 (自動封門)
    kernel_close = np.ones((7, 7), np.uint8) 
    kernel_open = np.ones((3, 3), np.uint8)  
    
    processed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open, iterations=2)
    processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel_close, iterations=3)
    
    # 4. 找輪廓
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rooms_data = []
    
    # 如果有指定高亮房間，先將整張圖變暗 (Dimming Effect)
    # highlight_id 為 None 或 "顯示全部" 時不變暗
    output_img = original.copy()
    if highlight_id and highlight_id != "顯示全部 (Show All)":
        # 創建黑色遮罩
        overlay = np.zeros_like(original)
        # 原圖變暗
        output_img = cv2.addWeighted(original, 0.4, overlay, 0.6, 0)
    
    room_id = 1
    
    for cnt in contours:
        area_px = cv2.contourArea(cnt)
        
        # 轉換為平方米
        if scale > 0:
            area_m2 = area_px / (scale ** 2)
        else:
            area_m2 = 0
            
        # 過濾雜訊
        if area_m2 > min_area_m2:
            current_id = f"R{room_id}"
            
            # --- 判斷是否需要繪製 ---
            is_highlighted = (highlight_id == current_id)
            show_all = (highlight_id is None or highlight_id == "顯示全部 (Show All)")
            
            # 只有在「顯示全部」或「選中該房間」時才畫
            if show_all or is_highlighted:
                # 顏色設定：選中用紅色，普通用綠色
                color = (0, 0, 255) if is_highlighted else (0, 255, 0)
                # 線條粗細：選中時加粗
                thickness = dynamic_thickness * 3 if is_highlighted else dynamic_thickness
                
                # 畫輪廓
                cv2.drawContours(output_img, [cnt], -1, color, thickness)
                
                # 標註中心點文字
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    
                    # 標籤內容
                    label = current_id
                    if is_highlighted:
                        label += f" ({area_m2:.1f}m2)" # 高亮時顯示面積詳細
                    
                    # 計算文字大小以便畫背景框
                    (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, dynamic_font_scale, dynamic_thickness)
                    
                    # 畫白色背景框 (確保文字清晰)
                    cv2.rectangle(output_img, (cX, cY - text_h - 10), (cX + text_w, cY + 10), (255, 255, 255), -1)
                    # 畫文字
                    cv2.putText(output_img, label, (cX, cY), 
                               cv2.FONT_HERSHEY_SIMPLEX, dynamic_font_scale, color, dynamic_thickness)
            
            # 收集數據 (不管有沒有畫出來，數據都要有)
            rooms_data.append({
                "Room ID": current_id,
                "Area (m²)": round(area_m2, 3),
            })
            room_id += 1
            
    return output_img, rooms_data, processed

# --- 主界面邏輯 ---

if uploaded_file is not None:
    # 讀取圖片
    image = Image.open(uploaded_file)
    
    # 1. 預先執行一次分析，只為了獲取房間列表 (供 Dropdown 使用)
    _, initial_data, _ = process_image(image, threshold_val, blur_kernel, scale_ratio, min_area_filter)
    
    # 提取所有房間 ID
    room_options = ["顯示全部 (Show All)"]
    if initial_data:
        room_list = [item["Room ID"] for item in initial_data]
        room_options += room_list

    # 2. 顯示房間搜尋器
    st.markdown("### 🔍 房間定位器 (Room Locator)")
    col_search, col_info = st.columns([1, 2])
    with col_search:
        selected_room = st.selectbox("選擇你想查看的房間 ID：", room_options)
    
    # 3. 再次執行分析 (帶著高亮參數)
    result_img, data, debug_img = process_image(image, threshold_val, blur_kernel, scale_ratio, min_area_filter, highlight_id=selected_room)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("原始圖則")
        # 修正：使用 use_container_width 代替舊參數
        st.image(image, use_container_width=True)
        
    with col2:
        st.subheader("AI 識別結果")
        # 修正：使用 use_container_width 代替舊參數
        st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), use_container_width=True)

    # --- 互動式 GFA 計算表 ---
    st.subheader("📊 智能面積表 (Interactive GFA Schedule)")

    if data:
        df = pd.DataFrame(data)
        
        # 初始化欄位
        if "Exempt? (豁免)" not in df.columns:
            df.insert(0, "Exempt? (豁免)", False)
        if "Room Name" not in df.columns:
            df.insert(1, "Room Name", df["Room ID"])
        df["Remarks"] = ""

        st.info("💡 **AP 小貼士**：你可以直接在表格中修改「房間名稱」或勾選「豁免」(如機房/露台)，總 GFA 會自動扣除！")
        
        # 顯示 Data Editor
        edited_df = st.data_editor(
            df,
            column_config={
                "Exempt? (豁免)": st.column_config.CheckboxColumn(
                    "豁免 GFA?",
                    help="勾選後，此房間將不會計入總 GFA",
                    default=False,
                ),
                "Room Name": st.column_config.TextColumn(
                    "房間名稱",
                    help="點擊修改",
                    required=True,
                ),
                "Area (m²)": st.column_config.NumberColumn(
                    "面積 (m²)",
                    format="%.3f",
                    disabled=True,
                ),
            },
            hide_index=True,
            use_container_width=True,
            num_rows="dynamic"
        )

        # GFA 計算邏輯
        gfa_df = edited_df[edited_df["Exempt? (豁免)"] == False]
        exempt_df = edited_df[edited_df["Exempt? (豁免)"] == True]

        total_area = edited_df["Area (m²)"].sum()
        total_gfa = gfa_df["Area (m²)"].sum()
        exempted_area = exempt_df["Area (m²)"].sum()

        st.write("---")
        m1, m2, m3 = st.columns(3)
        
        with m1:
            st.metric(label="總建築面積 (Construction Area)", value=f"{total_area:.3f} m²")
        with m2:
            st.metric(label="❌ 豁免面積 (Exempted)", value=f"{exempted_area:.3f} m²", delta=f"-{exempted_area:.3f}")
        with m3:
            st.metric(label="✅ **總樓面面積 (Total GFA)**", value=f"**{total_gfa:.3f} m²**", delta_color="normal")

        # 下載 CSV
        csv = edited_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 下載 GFA 計算表 (CSV)",
            data=csv,
            file_name="GFA_Schedule_Calculated.csv",
            mime="text/csv",
        )

    else:
        st.warning("⚠️ 找不到房間，請調整 Threshold 或 Blur 參數。")

    # Debug 視圖
    with st.expander("🔧 開發者視圖 (Debug View - 二值化影像)"):
        if 'debug_img' in locals():
            st.image(debug_img, use_container_width=True, channels="GRAY")

else:
    st.info("👈 請在左側上傳一張平面圖開始測試。")
