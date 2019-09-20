#!/usr/bin/env python
# coding: utf-8

# In[1]:


pwd


# In[2]:


from pathlib import Path


# In[3]:


data_top_level_dir = Path( '../first_batch')


# In[4]:


# Create directories if they don't already exist
output_dir = Path( '../pipeline_output')

if not output_dir.exists():
    print( f"Making directory {output_dir.resolve()}")
    output_dir.mkdir()

bg_image_output_dir = output_dir / "bg_images"
concat_vid_output_dir = output_dir/ "concatenated_videos"
timestamp_output_dir = output_dir / "timestamp_datadump"

for subdir_path in [ bg_image_output_dir, concat_vid_output_dir, timestamp_output_dir]:
    if not subdir_path.exists():
        print( f"Making directory {subdir.resolve()}")
        subdir.mkdir()


# In[5]:


# assemble inputs and outputs and check for existing files
potential_targets = list( data_top_level_dir.glob("*/*.rec") )
filedata = []
for recfile in potential_targets:
    
    stem = recfile.stem

    relevant_paths = [
        bg_image_output_dir / f"{stem}_BACKGROUND.tif",
        concat_vid_output_dir / f"{stem}_CONCATENATED.tif",
        timestamp_output_dir / f"{stem}_TIMESTAMP_PLOT.pdf",
        timestamp_output_dir / f"{stem}_TIMESTAMP_DATA.csv"
    ]
    
    have_files = [ _.exists() for _ in relevant_paths ]
    if all( have_files ):
        print( f"Skipping {stem} since everything is already done.")
        continue
    
    for p, h in zip( relevant_paths, have_files ):
        print( p, h )
    
    video_parts_paths = sorted( list( recfile.parent.glob( recfile.stem + "*.tif" ) ) )
    if len( video_parts_paths ) == 0:
        print( f'ERROR: didn\'t find any videos with prefix "{input_vid_fname_stem}" in directory "{input_dir.resolve()}"')
        continue

    filedata.append( ( recfile.stem, recfile.parent.name, video_parts_paths, *relevant_paths ) )

print( f"Found {len(potential_targets)} potential processing targets, processing {len(filedata)}." )


# In[6]:


def DumpTimestamps( video, vid_boundary_frames, output_fig_path, output_csv_path, input_vid_fname_stem, debug=True ):
    """Takes video as input and dumps hex-encoded timestamps to file"""

    import pandas as pd
    import numpy as np
    all_timestamps = []
    for pixels in video[ :, 0, :14 ]:
        # convert pixel integers to strings with hexadecimal representation
        # eg integer 1 output is '0x1'
        # crop the '0x' off the string, and pad each digit with leading zeros if necessary.
        # Then join all the 2-character strings together into one big string.
        all_timestamps.append( "".join( [ hex(_)[2:].zfill(2) for _ in pixels ] ) )

    df = pd.DataFrame( all_timestamps, columns=['raw'] )

    def FormatTimestamp( s ):
        tstr = list( s )
        tstr.insert( 8, ' ' )
        tstr.insert( 13, '-' )
        tstr.insert( 16, '-' )
        tstr.insert( 19, ' ' )
        tstr.insert( 22, ':' )
        tstr.insert( 25, ':' )
        tstr.insert( 28, '.' )
        return "".join( tstr )

    # Assign each frame in the concatenated video the video part it came from
    df['video'] = 1
    for i, frame_i in enumerate( vid_boundary_frames, start=2 ):
        rows = list( range( frame_i, len( video ) ) )
        df.loc[ rows, 'video' ] = i

    # Parse timestamp
    df['timestamp'] = df['raw'].apply( FormatTimestamp )
    df['frame_index'] = df['raw'].str[:8]
    df['date'] = df['raw'].str[8:16]
    df['hour'] = df['raw'].str[16:18]
    df['min'] = df['raw'].str[18:20]
    df['rawsec'] = df['raw'].str[20:22]

    df['sec'] = pd.to_numeric( df['rawsec'], errors='coerce' ).fillna( method='ffill').astype(int)

    # Adjust for seconds rolling over to the next minute
    sec_copy = df['sec'].values.copy()
    t_i = sec_copy[0]
    for i, t_i_plus_1 in enumerate( df['sec'].values[1:], start=1):
        if t_i_plus_1 < t_i:
            sec_copy[i] += 60
    df['sec'] = sec_copy
    df['sec'] = df['sec'].astype(str)

    df['sec_fraction'] = df['raw'].str[22:].str.extract( r'^(\d+)' )#.str.ljust(6,'0')
    df['raw_time'] = pd.to_numeric( df['sec']  + '.' + df['sec_fraction'], errors='coerce' ).fillna( method='bfill')

    #from scipy.stats import linregress
    #slope, intercept, r_value, p_value, std_err = linregress( np.array( df.index ), df['time'].values )
    #from sklearn.linear_model import RANSACRegressor
    #model = RANSACRegressor()
    from sklearn.linear_model import TheilSenRegressor
    model = TheilSenRegressor()
    X = np.array( df.index ).reshape(-1, 1)
    Y = df['raw_time'].values
    model.fit( X, Y )
    print( "Fitting timestamps with slope and intercept.")
    print( f"Timestamp estimated coefficients: intercept={float(model.intercept_):0.2f}, slope={float(model.coef_)*1000:0.3f}ms/frame" )

    Y_pred = model.predict( X )
    df['adj_time'] = Y_pred
    df['delta t (ms)'] = (df['raw_time'] - df['raw_time'].shift() ).fillna(0) * 1000
    df['delta t %-ile'] = df['delta t (ms)'].rank( pct=True )

    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots( dpi=300 )
    df.plot( y=['raw_time', 'adj_time'], ax=ax1 )
    ax1.set_ylabel( "time (s)" )
    ax1.set_xlabel( "frame index" )
    ax1.set_title( f'"{input_vid_fname_stem}" raw and adjusted timestamps' )
    #ax2 = ax1.twinx()
    #ax2.plot( (out['adj_time']-out['reg_time']), label='diff', color='r')
    #ax2.set_ylabel( "difference between raw time and straight line (s)")
    for x in vid_boundary_frames:
        ax1.axvline( x, linestyle='dashed', color='black' )

    fig.savefig( str( output_fig_path ) )
    plt.close(fig) # Explicitly close to free memory and avoid warning
    df.to_csv( str( output_csv_path ) )
    print( f"Wrote \"{str( output_fig_path )}\" and \"{str( output_csv_path ) }\" to disk" )

    return (df.loc[len(video)-1,'adj_time' ] - df.loc[0,'adj_time' ])


# In[7]:


def ConcatenateVideos( input_vid_fname_stem, input_dir, input_video_paths, 
                      bg_img_output_path, output_vid_path,
                      timestamp_output_fig_path, timestamp_output_csv_path,
                      lower_clip_pctile=2.5, upper_clip_pctile=97.5, debug=True):

    import time
    t00 = time.time()
    from skimage.io import imread, imsave
    import numpy as np
    from pathlib import Path

    if debug:
        print( '\n\n******************************************' )
        print( 'TODO:')
        print( f'Processing "{input_vid_fname_stem}"' )
        print( f'From directory "{input_dir}"' )
        print( f'Concatenate these video parts: {input_video_paths}' )
        print( f'Derive background image and save to "{bg_img_output_path}"' )
        print( f'Save concatenated contrast adjusted video to "{output_vid_path}"' )
        print( f'Dump timestamp data to "{timestamp_output_csv_path}"' )
        print( f'Create timestamp plot in "{timestamp_output_fig_path}"' )
        print( "beginning processing now...\n" )
        
        
    # ========================================================
    # Part 2: Load videos and concatenate
    if debug:
        t0 = time.time()
    
    # imread/imsave doesn't play well with pathlib.Path objects, convert to str!
    all_vids = [ imread(str(_)) for _ in input_video_paths ]
    all_vid_nframes = [ _.shape[0] for _ in all_vids ]
    all_vid_nframes = all_vid_nframes[:] # deep copy so that the intermediate videos have no reference to them and get deleted
    out_im = np.concatenate( all_vids )
    del all_vids
    
    if debug:
        t1 = time.time()
        print( f"Loading source videos (shape={out_im.shape}, {out_im.nbytes/1073741824:0.3f} GB total) took {t1-t0:0.2f} seconds" )

    # ========================================================
    # Part 3: Dump timestamps if they don't exist already
    if timestamp_output_csv_path.exists() and timestamp_output_fig_path.exists():
        print( f'Skipping timestamp processing for {input_vid_fname_stem} since "{str(timestamp_output_csv_path)}" and "{str(timestamp_output_fig_path)}" already exist.')
    else:
        if debug:
            t0 = time.time()

        from itertools import accumulate
        vid_boundary_frames = list( accumulate( all_vid_nframes ) )[:-1]

        kwargs = { 
            "video" : out_im,
            "vid_boundary_frames" : vid_boundary_frames,
            "output_fig_path" : timestamp_output_fig_path,
            "output_csv_path" : timestamp_output_csv_path,
            "input_vid_fname_stem" : input_vid_fname_stem,
            "debug" : debug }

        elapsed_time = DumpTimestamps( **kwargs )
        print( f"Total video elapsed time {elapsed_time:0.2f}s over {len(out_im)} frames={elapsed_time/len(out_im)*1000:0.2f} miliseconds per frame" )

        if debug:
            t1 = time.time()
            print( f"Timestamp processing took {t1-t0:0.2f} seconds" )
    
    
    # Create a bail point just in case you had wanted only to regenerate the timestamps
    if bg_img_output_path.exists() and output_vid_path.exists():
        print( f"Skipping writing {str(bg_img_output_path)} and {str(output_vid_path)} since they already exist." )
        return
    
    # ========================================================
    # Part 4a: Calculate upper and lower percentiles for clipping purposes
    if debug:
        t0 = time.time()

    wanted_px_pctiles = [ 0, lower_clip_pctile, upper_clip_pctile, 100 ]
    px_pctiles = np.percentile( out_im.flat, wanted_px_pctiles )
    # Amazingly, the return type of the clipped matrix depends on the type of the lbound
    # and ubound. I want integer output, but the output from np.percentile are floats.
    pixmin, lbound, ubound, pixmax = [ int(round(_)) for _ in px_pctiles ]
    if debug:
        t1 = time.time()
        print( f"Calculating pixel intensity {lower_clip_pctile}%ile and {upper_clip_pctile}%ile for \"{input_vid_fname_stem}\" took {t1-t0:0.2f} seconds" )

    # ========================================================
    # Part 4b: Clip pixels and perform min/max normalization to 0-255
    if debug:
        t0 = time.time()

    np.clip( out_im, lbound, ubound, out=out_im )
    out_im -= lbound
    out_im = ( out_im.astype( np.float32 ) * (255/(ubound-lbound)) ).round().astype( np.uint8 )

    if debug:
        t1 = time.time()
        print( f"Clipping pixel range from {pixmin}-{pixmax} ({pixmax-pixmin} range) to {lbound}-{ubound} ({ubound-lbound} range) took {t1-t0:0.2f} seconds" )

    # ========================================================
    # Part 5: Generate background and save
    if debug:
        t0 = time.time()

    background_im = np.median( out_im, axis=0 ).round().astype( np.uint8 )
    imsave( str(bg_img_output_path), background_im )
    if debug:
        t1 = time.time()
        print( f"Calculating background for { input_vid_fname_stem } and saving to {bg_img_output_path} took {t1-t0:0.2f} seconds" )            

    # ========================================================
    # Part 6: Save concatenated video to disk
    if debug:
        t0 = time.time()

    imsave( str(output_vid_path), out_im )

    if debug:
        t1 = time.time()
        print( f"Saving { output_vid_path } ({out_im.nbytes/1073741824:0.3f} GB total) to disk took {t1-t0:0.3f} seconds" )

    print( f"Processing \"{input_vid_fname_stem}\" took {time.time()-t00:0.2f} seconds")


# In[8]:


for row in filedata:
    ConcatenateVideos( *row )


# In[ ]:




#!/usr/bin/env python
# coding: utf-8

# In[1]:


pwd


# In[8]:


from pathlib import Path


# In[9]:


data_dir = Path( '../pipeline_output').resolve()
input_videos_dir = data_dir / "concatenated_videos"
input_backgrounds_dir = data_dir / "bg_images"
output_dir = data_dir / "bg_subtr_videos"


# In[10]:


if not output_dir.exists():
    print( f"Making directory {output_dir.resolve()}")
    subdir.mkdir()


# In[11]:


# Rebuild file list to figure out what's next
input_data = []
#print( "****************************")
input_video_paths = list(input_videos_dir.glob("*_CONCATENATED.tif"))
for input_video_path in input_video_paths:

    stem = input_video_path.name[:-17]
    #print( stem )
    
    # Already done?
    output_path = output_dir / f"{stem}_BG_SUBTR.tif"
    if output_path.exists():
        print( f"Skipping \"{str(output_path)}, since it already exists.")
        continue

    bg_img_path = input_backgrounds_dir / f"{stem}_BACKGROUND.tif"
    if not bg_img_path.exists():
        print( f"WARNING: {str(bg_img_path)} doesn't exist, please regenerate.")
        continue

    input_data.append( ( input_video_path, bg_img_path, output_path ) )
    
print( f"Found {len(input_video_paths)} vidoes, processing {len( input_data) } of them." )


# In[12]:


def BackgroundSubtract( input_video_path: Path, bg_img_path: Path,
                       output_path: Path,
                       lower_clip_pctile=2.5, upper_clip_pctile=97.5, debug=True ):

    if output_path.exists():
        print( f'Skipping processing "{str(output_path)}" since file already exists.')
        return

    import time
    if debug:
        print( "\n\n****************************")
        print( "TODO:")
        print( "Input video path:", str( input_video_path ) )
        print( "Background image path:", str( bg_img_path ) )
        print( "Output video path:", str( output_path ) )
        print( "beginning processing now...\n" )
        t0 = time.time()
        t00 = t0

    # ========================================================
    # Part 1: Load video from disk and subtract backgound
    from skimage.io import imread, imsave
    import numpy as np
    
    video = imread( str(input_video_path) ).astype( np.float32 )
    bg = imread( str(bg_img_path) ).astype( np.float32 )
    video -= bg
    if debug:
        t1 = time.time()
        print( f"Loaded source video, {video.nbytes/1073741824:0.3f} GB total after converting datatype from int8 to float32. Took {t1-t0:0.2f} seconds" )

    # ========================================================
    # Part 2a: Calculate upper and lower percentiles for clipping purposes
    if debug:
        t0 = time.time()

    wanted_px_pctiles = [ 0, lower_clip_pctile, upper_clip_pctile, 100 ]
    px_pctiles = np.percentile( video.flat, wanted_px_pctiles )
    pixmin, lbound, ubound, pixmax = [ int(round(_)) for _ in px_pctiles ]
    
    if debug:
        t1 = time.time()
        print( f"Calculating pixel intensity {lower_clip_pctile}%ile and {upper_clip_pctile}%ile took {t1-t0:0.2f} seconds" )
    
    # ========================================================
    # Part 2b: Clip pixels and perform min/max normalization to 0-255
    if debug:
        t0 = time.time()

    np.clip( video, lbound, ubound, out=video )
    video -= lbound
    try:
        video = ( video * (255/(ubound-lbound)) ).round().astype( np.uint8 )
    except ZeroDivisionError:
        print( f"ERROR: Something funky with video \"{str(input_video_path)}\"")
        print( f"pixmin={pixmin}, {lower_clip_pctile}%ile={lbound}, {upper_clip_pctile}%ile={ubound}, pixmax={pixmax}")
        return

    if debug:
        t1 = time.time()
        print( f"Clipping pixel range from {pixmin}-{pixmax} ({pixmax-pixmin} range) to {lbound}-{ubound} ({ubound-lbound} range) took {t1-t0:0.2f} seconds" )

    # ========================================================
    # Part 3: Save
    if debug:
        t0 = time.time()
        
    imsave( str(output_path), video )
    
    if debug:
        t1 = time.time()
        print( f"Saving { str(output_path) } ({video.nbytes/1073741824:0.3f} GB total) to disk took {t1-t0:0.3f} seconds" )

    print( f"TOTAL PROCESSING TIME for \"{input_video_path}\" took {time.time()-t00:0.2f} seconds.")


# In[13]:


for paths in input_data:
    BackgroundSubtract( *paths )


# In[ ]:




