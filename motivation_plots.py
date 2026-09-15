# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import binned_statistic_2d, linregress
# from pathlib import Path
# import rasterio
# import logging

# logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
# log = logging.getLogger(__name__)

# # Paths
# DATA_DIR = Path("data/era5_processed")
# PRISM_DIR = Path("data/prism_tif_2018_2024")
# PRISM_DIR_2025 = Path("data/prism_tif_2025/2025")
# OUT_DIR = Path("results/motivation")
# OUT_DIR.mkdir(parents=True, exist_ok=True)

# def main():
#     # 1. Load ERA5 Metadata and Grid
#     log.info("Loading ERA5 metadata...")
#     era5_lats = np.load(DATA_DIR / "era5_lats.npy")
#     era5_lons = np.load(DATA_DIR / "era5_lons.npy")

#     lat_sort_idx = np.argsort(era5_lats)
#     lon_sort_idx = np.argsort(era5_lons)

#     era5_lats = era5_lats[lat_sort_idx]
#     era5_lons = era5_lons[lon_sort_idx]
    
#     print(era5_lats[:3], era5_lats[-3:])
#     print(era5_lons[:3], era5_lons[-3:])
#     # Standardize Longitudes to -180 to 180
#     if era5_lons.max() > 180:
#         log.info("Converting ERA5 lons from 0-360 to -180-180 format")
#         era5_lons = (era5_lons + 180) % 360 - 180
    
#     # Ensure they are negative for the US
#     if np.all(era5_lons > 0):
#         log.info("Flipping positive Western longitudes to negative.")
#         era5_lons = -era5_lons

#     era5_precip_all = np.load(DATA_DIR / "pair_A_fine.npy")
    
#     # Standardize ERA5 units to mm (if input is meters)
#     if era5_precip_all.max() < 1.0:
#         log.info("Converting ERA5 from meters to mm")
#         era5_precip_all = era5_precip_all * 1000.0
        
#     era5_dates = np.load(DATA_DIR / "valid_times_A.npy", allow_pickle=True)
#     test_idx = np.load(DATA_DIR / "test_indices_A.npy")

#     # Setup bin edges (scipy requires strictly increasing)
#     dlat, dlon = 0.25, 0.25
#     lat_edges = era5_lats - dlat/2
#     lat_edges = np.append(lat_edges, lat_edges.max() + dlat)
#     lon_edges = era5_lons - dlon/2
#     lon_edges = np.append(lon_edges, lon_edges.max() + dlon)
    
#     all_era5 = []
#     all_prism = []

#     # 2. Process each test day
#     log.info(f"Processing {len(test_idx)} test days...")

#     for i in test_idx:
        
#         date_obj = era5_dates[i]
#         date_str = str(date_obj)[:10].replace("-", "") 
#         # log.info(f"Day index {i} corresponds to date {date_str}")
#         year = date_str[:4]

#         # Robust Pathing
#         target_file = f"prism_ppt_us_30s_{date_str}.tif"
#         if year == "2025":
#             full_path = PRISM_DIR_2025 / target_file
#         else:
#             full_path = PRISM_DIR / year / target_file
        
#         if not full_path.exists():
#             continue
            
#         with rasterio.open(full_path) as src:
#             p_data = src.read(1).astype(np.float32)
#             p_data[p_data < 0] = 0
            
#             transform = src.transform
#             cols, rows = np.meshgrid(np.arange(src.width), np.arange(src.height))
#             p_lons, p_lats = rasterio.transform.xy(transform, rows, cols)
            
#             # Binning
#             ret = binned_statistic_2d(
#                 np.array(p_lats).flatten(), np.array(p_lons).flatten(), p_data.flatten(),
#                 statistic='mean', bins=[lat_edges, lon_edges]
#             )
#             prism_aggregated = ret.statistic
            
#             # ERA5 Orientation Correction
#             era5_vals = era5_precip_all[i]
            
#             # 1. Shape check (Transpose if Lon/Lat order is swapped)
#             # if era5_vals.shape != prism_aggregated.shape:
#             #     era5_vals = era5_vals.T
            
#             # 2. THE CRITICAL FIX: Flip Latitude (North-South vs South-North)
#             # era5_vals = np.flipud(era5_vals)
#             # era5_vals = np.flipud(np.fliplr(era5_vals))
#             # era5_vals = np.flipud(era5_vals)   # latitude
#             # era5_vals = np.fliplr(era5_vals)   # longitude
#             # if era5_vals.shape == (len(era5_lons), len(era5_lats)):
#             #     era5_vals = era5_vals.T

#             # Flip latitude if needed
#             # if era5_lats[0] > era5_lats[-1]:
#             #     era5_vals = np.flipud(era5_vals)

#             # # Flip longitude if needed
#             # if era5_lons[0] > era5_lons[-1]:
#             # era5_vals = np.fliplr(era5_vals)
#             era5_vals = era5_vals[lat_sort_idx, :]
#             era5_vals = era5_vals[:, lon_sort_idx]
            
#             # if i == test_idx[0]:
#             #     plt.figure(figsize=(10, 4))
#             #     plt.subplot(1, 2, 1)
#             #     plt.imshow(era5_vals, origin='lower')
#             #     plt.title("ERA5 Grid")
#             #     plt.subplot(1, 2, 2)
#             #     plt.imshow(prism_aggregated, origin='lower')
#             #     plt.title("PRISM Aggregated")
#             #     plt.savefig(OUT_DIR / "debug_map_alignment.png")
#             #     log.info("Saved debug maps. Please check if the rain patterns look shifted or flipped.")
#             # mask = ~np.isnan(prism_aggregated)
#             mask = (~np.isnan(prism_aggregated) & ~np.isnan(era5_vals))
#             all_era5.extend(era5_vals[mask].flatten())
#             all_prism.extend(prism_aggregated[mask].flatten())

#     # 3. Post-Processing and Scaling
#     if len(all_era5) == 0:
#         log.error("Collected 0 points. Check your coordinates!")
#         return

#     x = np.array(all_era5)
#     y = np.array(all_prism)
    
#     # Calculate scaling factor based on non-zero rain events
#     rain_mask = (x > 0.1) & (y > 0.1)
#     if np.any(rain_mask):
#         scaling_factor = np.mean(y[rain_mask]) / np.mean(x[rain_mask])
#     else:
#         scaling_factor = y.max() / x.max()
        
#     log.info(f"Applying scaling factor: {scaling_factor:.2f}")
#     x_scaled = x * scaling_factor

#     # Filter for the plot (Rainy days only to show the relationship)
#     plot_mask = (x_scaled > 1.0) & (y > 1.0)
#     x_plot = x_scaled[plot_mask]
#     y_plot = y[plot_mask]

#     if len(x_plot) > 0:
#         slope, intercept, r_val, p_val, std_err = linregress(x_plot, y_plot)
#         bias = np.mean(y_plot - x_plot)
#         nmb = (bias / np.mean(x_plot)) * 100
#     else:
#         log.error("No rainy points for regression.")
#         return

#     # 4. Plotting
#     plt.figure(figsize=(7, 6))
#     plt.scatter(x_plot, y_plot, s=2, alpha=0.15, color='#0072B2', label='Daily Grid Cells (>1mm)')
    
#     limit = max(x_plot.max(), y_plot.max())
#     plt.plot([0, limit], [0, limit], 'k--', linewidth=1.5, label='1:1 Line')
#     plt.plot(x_plot, slope * x_plot + intercept, 'r-', linewidth=1.5, label='Linear Fit')

#     stats_text = f"Slope: {slope:.2f}  Int: {intercept:.2f}  Cor: {r_val:.2f}  Bias: {bias:.2f}  NMB: {nmb:.1f}%"
#     plt.title(f"ERA5 (25km, Scaled) vs PRISM Aggregated Mean\n{stats_text}", fontsize=9)
#     plt.xlabel("ERA5 Precipitation (mm/day)")
#     plt.ylabel("PRISM Precipitation (mm/day)")
#     plt.grid(alpha=0.3)
#     plt.legend()
#     plt.xlim(0, limit)
#     plt.ylim(0, limit)
#     plt.savefig(OUT_DIR / "motivation_scatter.png", dpi=300)
#     log.info(f"Complete. Plot saved to {OUT_DIR}")

# if __name__ == "__main__":
#     main()

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d, linregress
from pathlib import Path
import rasterio
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Paths
DATA_DIR = Path("data/era5_processed")
PRISM_DIR = Path("data/prism_tif_2018_2024")
PRISM_DIR_2025 = Path("data/prism_tif_2025/2025")
OUT_DIR = Path("results/motivation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    # 1. Load Metadata
    log.info("Loading ERA5 metadata...")
    era5_lats_raw = np.load(DATA_DIR / "era5_lats.npy")
    era5_lons_raw = np.load(DATA_DIR / "era5_lons.npy")
    era5_precip_all = np.load(DATA_DIR / "pair_A_fine.npy")
    era5_dates = np.load(DATA_DIR / "valid_times_A.npy", allow_pickle=True)
    test_idx = np.load(DATA_DIR / "test_indices_A.npy")

    # 2. Harmonize Coordinates
    # Standardize to -180/180
    if era5_lons_raw.max() > 180:
        era5_lons_raw = (era5_lons_raw + 180) % 360 - 180
    if np.all(era5_lons_raw > 0):
        era5_lons_raw = -era5_lons_raw

    # Sort for Monotonic Bin Edges (Required by Scipy)
    lat_sort_idx = np.argsort(era5_lats_raw)
    lon_sort_idx = np.argsort(era5_lons_raw)
    era5_lats = era5_lats_raw[lat_sort_idx]
    era5_lons = era5_lons_raw[lon_sort_idx]

    # Setup bin edges
    dlat, dlon = 0.25, 0.25
    lat_edges = np.sort(era5_lats - dlat/2)
    lat_edges = np.append(lat_edges, lat_edges.max() + dlat)
    lon_edges = np.sort(era5_lons - dlon/2)
    lon_edges = np.append(lon_edges, lon_edges.max() + dlon)
    
    all_era5 = []
    all_prism = []
    total_days = era5_precip_all.shape[0]
    # 3. Process each test day
    # log.info(f"Processing {total_days} test days...")
    for i in range(total_days):
        date_obj = era5_dates[i]
        # --- THE FIX: ADD ONE DAY ---
        # If ERA5 represents the 24h leading up to 00Z, 
        # it likely matches the PRISM file dated for the NEXT day.
        offset_date = date_obj + np.timedelta64(1, 'D')
        date_str = str(offset_date)[:10].replace("-", "") 
        year = date_str[:4]
        

        # Pathing logic
        target_file = f"prism_ppt_us_30s_{date_str}.tif"
        full_path = (PRISM_DIR_2025 / target_file) if year == "2025" else (PRISM_DIR / year / target_file)
        
        if not full_path.exists():
            continue
        try:
            with rasterio.open(full_path) as src:
                p_data = src.read(1).astype(np.float32)
                p_data[p_data < 0] = 0 # Mask missing/negative
                
                # Generate PRISM coordinate grid
                transform = src.transform
                cols, rows = np.meshgrid(np.arange(src.width), np.arange(src.height))
                p_lons, p_lats = rasterio.transform.xy(transform, rows, cols)
                
                # Aggregate PRISM pixels to ERA5 cells
                ret = binned_statistic_2d(
                    np.array(p_lats).flatten(), np.array(p_lons).flatten(), p_data.flatten(),
                    statistic='mean', bins=[lat_edges, lon_edges]
                )
                prism_agg = ret.statistic # (Lat, Lon) grid matching sorted edges

                # Extract and Align ERA5 slice
                # We apply the same sort to the data that we did to the edges
                era5_slice = era5_precip_all[i]
                era5_aligned = era5_slice[lat_sort_idx, :][:, lon_sort_idx]
                
                # If ERA5 was stored North-to-South, the sort fixed it. 
                # But we must check if it's still 'flipped' relative to Scipy's binning.
                # Based on your debug maps, we apply the vertical flip to align indices.
                era5_aligned = np.flipud(era5_aligned)

                # Standardize Units (m -> mm)
                if era5_aligned.max() < 1.0:
                    era5_aligned *= 1000.0

                # 4. Explicit Index Pairing
                # Avoids flattening errors by iterating through the lat/lon grid directly
                r_limit, c_limit = prism_agg.shape
                for r in range(r_limit):
                    for c in range(c_limit):
                        p_val = prism_agg[r, c]
                        e_val = era5_aligned[r, c]
                        if not np.isnan(p_val) and not np.isnan(e_val):
                            all_era5.append(e_val)
                            all_prism.append(p_val)

        except (rasterio.errors.RasterioIOError, Exception) as e:
            log.warning(f"Skipping corrupted or unreadable file {full_path.name}: {e}")
            continue

    # 5. Final Statistics
    if not all_era5:
        log.error("Zero overlap found.")
        return

    x, y = np.array(all_era5), np.array(all_prism)
    
    # Scale to align intensities (Dynamic Denormalization)
    rain_mask = (x > 0.1) & (y > 0.1)
    scale = np.mean(y[rain_mask]) / np.mean(x[rain_mask]) if np.any(rain_mask) else 1.0
    x_scaled = x * scale

    # Plot Filter: Rainy events only
    plot_mask = (x_scaled > 1.0) & (y > 1.0)
    xp, yp = x_scaled[plot_mask], y[plot_mask]

    if xp.size > 0:
        slope, intercept, r_val, p_val, std_err = linregress(xp, yp)
        bias = np.mean(yp - xp)
        nmb = (bias / np.mean(xp)) * 100
    else:
        log.error("No rainy points for plot.")
        return

    # 6. Plotting
    plt.figure(figsize=(7, 6))
    plt.scatter(xp, yp, s=5, alpha=0.1, color='#0072B2', label='Daily Grid Cells')
    
    mx = max(xp.max(), yp.max())
    plt.plot([0, mx], [0, mx], 'k--', alpha=0.7, label='1:1 Line')
    plt.plot(xp, slope * xp + intercept, 'r-', label=f'Fit (m={slope:.2f})')

    plt.title(f"ERA5 vs PRISM Aggregated (NMB: {nmb:.1f}%, Cor: {r_val:.2f})")
    plt.xlabel("ERA5 (mm/day)")
    plt.ylabel("PRISM Mean (mm/day)")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.savefig(OUT_DIR / "motivation_scatter_final.png", dpi=300)
    log.info("Process complete.")

if __name__ == "__main__":
    main()