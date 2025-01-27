#!/usr/bin/env python3
# encoding: utf-8

import os
import argparse
import re
import matplotlib.pyplot as plt
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

# Constants
COLOR = '#60ADD9'
TEXT_COLOR = '#EBC326'
IMGH_DIRECTION_DEFAULT = 'vertical'
PRODUCE_IMAGES = True
UPPER_MARGIN = 0.45 # Percentage of the image covered by track

def parse_args():
    parser = argparse.ArgumentParser(description="Extract track data from a Wikiloc URL and overlay it on an image")
    parser.add_argument("image", type=str, help="Path to the image file.")
    parser.add_argument("url", type=str, help="URL to the Wikiloc track.")
    parser.add_argument("--plot_altitude", action="store_true", default=False, help="Plot altitude data")
    parser.add_argument("--plot_track_data", action="store_true", default=False, help="Plot track data")
    parser.add_argument("--img_direction", choices=['horizontal', 'vertical'], default=IMGH_DIRECTION_DEFAULT, help="Direction of the image layout")
    return parser.parse_args()

def fetch_dynamic_html(url):
    options = Options()
    service = Service("/usr/bin/chromedriver")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox") # linux only
    options.add_argument("--headless") 
    driver = webdriver.Chrome(service = service, options=options)
    driver.get(url)

    WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, "elevation-profile-svg")))

    html_content = driver.page_source
    driver.quit()
    return html_content

def extract_static_data(html):
    soup = BeautifulSoup(html, 'html.parser')
    
    def get_value_for_dt(dt_text):
        dt_tag = soup.find('dt', string=dt_text)  # Search <dt> por el texto
        if dt_tag:
            dd_tag = dt_tag.find_next('dd')
            if dd_tag:
                return dd_tag.get_text(strip=True)
        return None

    positive_elevation = get_value_for_dt("Desnivel positivo")
    max_altitude = get_value_for_dt("Altitud máxima")
    min_altitude = get_value_for_dt("Altitud mínima")
    distance = get_value_for_dt("Distancia")

    return {
        "positive_elevation": positive_elevation,
        "distance": distance,
        "min_altitude": min_altitude,
        "max_altitude": max_altitude,
    }

def extract_track_points(svg_content):
    # Use regex to extract points from the <polyline> tag
    match = re.search(r'points="([^"]+)"', svg_content)
    if match:
        points = match.group(1).split()
        points = [tuple(map(float, point.split(','))) for point in points]
        return points
    return []

def plot_track_data(image_path, track_points, plot_altitude, plot_track_data, img_direction, static_data):
  
    os.makedirs("./output", exist_ok = True)
    image_name, image_extension = os.path.splitext(os.path.basename(image_path))
    path = f"./output/{image_name}_out{image_extension}"
    path_track = f"./output/{image_name}_trackonly.png"
        
    dpi = 300

    fig, ax = plt.subplots(figsize=(20, 20))
    
    x = [point[0] for point in track_points[1:-1]]
    y = [point[1] for point in track_points[1:-1]]
    inverted_y = [np.max(y) + np.min(y) - point for point in y]
    
    # Track only plot
    # ax.fill_between(x, inverted_y, 0, color = COLOR, alpha = .75)
    # ax.plot(x, inverted_y, color=COLOR, linewidth=2)
    
    # ax.axis("off")
    # fig.patch.set_alpha(0)  # Fondo de la figura transparente
    # ax.set_facecolor((0, 0, 0, 0))  # Fondo del gráfico transparente
    # fig.savefig(path_track, dpi=dpi, transparent=True, bbox_inches="tight", pad_inches=0)
    # plt.close(fig)

    
    if PRODUCE_IMAGES:
        # Store over image
        image = plt.imread(image_path)
        
        # Figure has to be created with image sizes so resolution is not lost
        height, width, nbands = image.shape
        # print(width/height , height/width) # Check that proportion is the same
        figsize = (width / dpi, height / dpi)

        # Get pixel coordinates for current plot
        proportion = width / x[-1]
        upper_margin_value = height*(1-UPPER_MARGIN)
        scaled_x = [point*proportion for point in x]
        current_range_y = np.max(y) - np.min(y)
        # Use 2 thirds of space for track data
        y_scale_factor = max(1,(height*UPPER_MARGIN*0.6)/current_range_y)
        scaled_y = np.array(y)*y_scale_factor+upper_margin_value
        
        fig_image = plt.figure(figsize=figsize)
        ax = fig_image.add_axes([0, 0, 1, 1])
        ax.axis('off')
        ax.plot(scaled_x, scaled_y, c = COLOR, linewidth = 2)
        ax.fill_between(scaled_x, scaled_y, height, color = COLOR, alpha = .75)
        
        if plot_track_data:
            fontsize = int((height - np.min(scaled_y))/10+0.5)
            print(f"plot_track_data {fontsize = }")
            center_y = (np.min(scaled_y) + height)/2
            center_x = (np.max(scaled_x) + np.min(scaled_x))/2
            ax.text(center_x, center_y-fontsize*3, f"{static_data['distance']}", ha='center', fontsize=fontsize, color=TEXT_COLOR)
            ax.text(center_x, center_y+fontsize*2, f"Desnivel: {static_data['positive_elevation']}", ha='center', fontsize=fontsize, color=TEXT_COLOR)

        if plot_altitude:
            fontsize = int(max(fontsize/3, abs(np.max(scaled_y) - np.min(scaled_y))/5)+0.5)
            print(f"plot_altitude {fontsize = }")
            ax.text(5, np.min(scaled_y), f"Max {static_data['max_altitude']}", ha='left', fontsize=fontsize, color=TEXT_COLOR)
            ax.text(5, np.max(scaled_y), f"Min {static_data['min_altitude']}", ha='left', fontsize=fontsize, color=TEXT_COLOR)

        ax.imshow(image, interpolation='nearest') # Display the image.

        print(f"Store image in {path}")
        fig_image.savefig(path, dpi=dpi, transparent=True)



def main():
    
    args = parse_args()
    html_content = fetch_dynamic_html(args.url)
    static_data = extract_static_data(html_content)
    svg_match = re.search(r'<svg.*?elevation-profile-svg.*?>(.*?)</svg>', html_content, re.DOTALL)
    if svg_match:
        svg_content = svg_match.group(1)
        track_points = extract_track_points(svg_content)
    else:
        track_points = []

    plot_track_data(args.image, track_points, args.plot_altitude, args.plot_track_data, args.img_direction, static_data)

    print("Positive Elevation:", static_data["positive_elevation"])
    print("Total Distance:", static_data["distance"])
    print("Min Altitude:", static_data["min_altitude"])
    print("Max Altitude:", static_data["max_altitude"])

if __name__ == "__main__":
    main()
