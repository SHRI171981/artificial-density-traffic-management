import os
import time
import streamlit as st

# Import core stateful logic directly from the specified backend module
from scripts.density_backend import TrafficController

# Initialize global layout constraints
st.set_page_config(layout="wide", page_title="Autonomous Traffic Controller")

def render_lane(lane_id, is_green, physical, artificial, total, img_path):
    """
    Renders a miniaturized lane control panel with an integrated stateful traffic light.
    Utilizes HTML/CSS flexbsox for maximum vertical and horizontal space efficiency.
    """
    text_color = "#28a745" if is_green else "#dc3545"
    bg_color = "#e6ffe6" if is_green else "#ffe6e6"
    
    # Conditional color rendering for traffic light lenses
    red_light = "#ff0000" if not is_green else "#440000"
    yellow_light = "#444400"
    green_light = "#00ff00" if is_green else "#004400"

    # Inject compact flexbox layout and traffic light module
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; background-color: {bg_color}; padding: 4px; border-radius: 4px; border: 1px solid {text_color}; margin-bottom: 4px;">
            <div style="background-color: #222; padding: 4px; border-radius: 8px; display: flex; flex-direction: column; gap: 3px; margin-right: 8px;">
                <div style="width: 10px; height: 10px; border-radius: 50%; background-color: {red_light}; box-shadow: 0 0 2px {red_light};"></div>
                <div style="width: 10px; height: 10px; border-radius: 50%; background-color: {yellow_light};"></div>
                <div style="width: 10px; height: 10px; border-radius: 50%; background-color: {green_light}; box-shadow: 0 0 2px {green_light};"></div>
            </div>
            <div style="color: black; line-height: 1.1; font-size: 10px; width: 100%;">
                <div style="font-weight: bold; font-size: 12px; border-bottom: 1px solid #ccc; padding-bottom: 2px; margin-bottom: 2px;">
                    Lane {lane_id}
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span>P: {int(physical)}</span>
                    <span>A: {int(artificial)}</span>
                </div>
                <div style="font-weight: bold; margin-top: 2px;">
                    Total: {int(total)}
                </div>
            </div>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Retrieve and render the raw camera capture associated with the lane identifier
    valid_extensions = ['.jpg', '.jpeg', '.png']
    for ext in valid_extensions:
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
            break

# Primary execution interface
st.title("4-Way Autonomous Traffic Controller")
st.markdown("Real-time intersection monitoring using YOLO vision and starvation-prevention weights.")

if st.button("Start 25-Cycle Simulation", type="primary"):
    
    # Instantiate the backend controller with active simulation variance
    controller = TrafficController(alpha=1.0, sim_percent=10)

    # Establish an isolated container to manage cyclic UI redrawing
    dashboard_view = st.empty()

    for cycle in range(1, 26):
        # Execute backend sequence to return the current state matrix
        matrix, image_files = controller.evaluate_cycle()
        green_lane = matrix['green_lane']
        
        # Redraw the spatial grid layout
        with dashboard_view.container():
            st.subheader(f"Processing Cycle {cycle} of 25")
            st.divider()
            
            # SQUEEZE THE GRID: Use outer columns to force the intersection into the middle 50% of the screen
            spacer_left, center_grid, spacer_right = st.columns([1, 2, 1])
            
            with center_grid:
                # Row 1: North Axis (Lane A)
                r1_col1, r1_col2, r1_col3 = st.columns([1, 1, 1])
                with r1_col2:
                    render_lane("A", 
                                is_green=(green_lane == "A"), 
                                physical=matrix['actual_counts'].get("A", 0), 
                                artificial=matrix['applied_ad_weight'].get("A", 0), 
                                total=matrix['effective_totals'].get("A", 0),
                                img_path=image_files["A"])
                
                st.write("") 
                
                # Row 2: West Axis (Lane D) and East Axis (Lane B)
                r2_col1, r2_col2, r2_col3 = st.columns([1, 0.5, 1])
                with r2_col1:
                    render_lane("D", 
                                is_green=(green_lane == "D"), 
                                physical=matrix['actual_counts'].get("D", 0), 
                                artificial=matrix['applied_ad_weight'].get("D", 0), 
                                total=matrix['effective_totals'].get("D", 0),
                                img_path=image_files["D"])
                # with r2_col2:
                #     # Render center intersection marker scaled to the miniaturized layout
                #     st.markdown("<div style='height: 100%; display: flex; align-items: center; justify-content: center; text-align: center;'><h1 style='color: gray; font-size: 30px; margin: 0;'>X</h1></div>", unsafe_allow_html=True)
                with r2_col3:
                    render_lane("B", 
                                is_green=(green_lane == "B"), 
                                physical=matrix['actual_counts'].get("B", 0), 
                                artificial=matrix['applied_ad_weight'].get("B", 0), 
                                total=matrix['effective_totals'].get("B", 0),
                                img_path=image_files["B"])

                st.write("") 

                # Row 3: South Axis (Lane C)
                r3_col1, r3_col2, r3_col3 = st.columns([1, 1, 1])
                with r3_col2:
                    render_lane("C", 
                                is_green=(green_lane == "C"), 
                                physical=matrix['actual_counts'].get("C", 0), 
                                artificial=matrix['applied_ad_weight'].get("C", 0), 
                                total=matrix['effective_totals'].get("C", 0),
                                img_path=image_files["C"])
                
        # Suspend thread execution to ensure the matrix shift is visually perceptible
        time.sleep(2.5)
        
    st.success("Simulation Complete")