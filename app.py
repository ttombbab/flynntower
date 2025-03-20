from flask import Flask, render_template, request, jsonify, redirect, url_for ,  session
from flask_cors import CORS
import requests
import json
import os  
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style, quantity_support
plt.style.use(astropy_mpl_style)
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import get_body, SkyCoord, EarthLocation, AltAz, get_sun
import datetime
import matplotlib.colors as mcolors
import random
from matplotlib.lines import Line2D

app = Flask(__name__)
app.secret_key = "rad_flint_key_extreme"
GAME_DATA_FILE = "game_data.json"
CORS(app)
concho_bonito = EarthLocation(lat=34.543055555555554*u.deg, lon=-109.58749999999999*u.deg, height=390*u.m) #34.543055555555554, -109.58749999999999
utcoffset = -7*u.hour  
CONCHO_BONITO_LAT = 34.543055555555554
CONCHO_BONITO_LON = -109.58749999999999

# Ollama server details (on the alarm server Pi)
OLLAMA_BASE_URL = "http://localhost:11434" #"http://192.168.0.107:8080"  # Replace with the actual IP address of your alarm server Pi

def generate_text(prompt, model="tomchat-fugawy", base_url="http://localhost:11434"):  # Replace with your Ollama server address
    url = f"{base_url}/api/generate"
    headers = {'Content-Type': 'application/json'}  # Important: Set the Content-Type header
    data = {
        "model": model,
        "prompt": prompt,
        "stream":False,

    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))  # Use json.dumps()
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        return response.json()  # Parse the JSON response
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        if 'response' in locals(): # Corrected typo 'responce' to 'response'
            if response.status_code != 200:
                print(f"Status Code: {response.status_code}")
                print(f"Response Text: {response.text}") # Print the response from the server for debugging
        return None

def generate_stargazing_plot(time, direction, filename):
    delta_time = np.linspace(-3, 3, 100) * u.hour
    times = time + delta_time
    frame = AltAz(obstime=times, location=concho_bonito)

    sunaltazs = get_sun(times).transform_to(frame)
    moonaltazs = get_body('moon', times).transform_to(frame)

    bright_stars = [
        ('Sirius', SkyCoord.from_name('Sirius')), ('Canopus', SkyCoord.from_name('Canopus')),
        ('Alpha Centauri', SkyCoord.from_name('Alpha Centauri')), ('Arcturus', SkyCoord.from_name('Arcturus')),
        ('Vega', SkyCoord.from_name('Vega')), ('Capella', SkyCoord.from_name('Capella')),
        ('Rigel', SkyCoord.from_name('Rigel')), ('Procyon', SkyCoord.from_name('Procyon')),
        ('Betelgeuse', SkyCoord.from_name('Betelgeuse')), ('Achernar', SkyCoord.from_name('Achernar'))
    ]

    planet_names = ['Mercury', 'Venus', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
    planets = [(planet, get_body(planet, times)) for planet in planet_names]

    # Generate unique colors
    num_stars = len(bright_stars)
    num_planets = len(planets)
    star_colors = plt.cm.get_cmap('viridis', num_stars)
    planet_colors = plt.cm.get_cmap('plasma', num_planets)

    plt.figure(figsize=(10, 6))
    plt.plot(delta_time.value, sunaltazs.alt.value, color='red', label='Sun')
    plt.plot(delta_time.value, moonaltazs.alt.value, color='gray', ls='--', label='Moon')

    for i, (name, star) in enumerate(bright_stars):
        star_altazs = star.transform_to(frame)
        if direction == 'N':
            az_range = (0, 90)
        elif direction == 'S':
            az_range = (180, 270)
        elif direction == 'E':
            az_range = (90, 180)
        elif direction == 'W':
            az_range = (270, 360)

        alt_values = []
        az_values = []
        for altaz in star_altazs:
            if az_range[0] <= altaz.az.value <= az_range[1]:
                alt_values.append(altaz.alt.value)
                az_values.append(altaz.az.value)

        if alt_values:
            plt.scatter(delta_time.value[:len(alt_values)], alt_values, color=star_colors(i), label=name)

    for i, (name, planet) in enumerate(planets):
        planet_altazs = planet.transform_to(frame)
        if direction == 'N':
            az_range = (0, 90)
        elif direction == 'S':
            az_range = (180, 270)
        elif direction == 'E':
            az_range = (90, 180)
        elif direction == 'W':
            az_range = (270, 360)

        alt_values = []
        az_values = []
        for altaz in planet_altazs:
            if az_range[0] <= altaz.az.value <= az_range[1]:
                alt_values.append(altaz.alt.value)
                az_values.append(altaz.az.value)

        if alt_values:
            plt.scatter(delta_time.value[:len(alt_values)], alt_values, color=planet_colors(i), label=name)
            
    if "sunset" in filename:
        plt.fill_between(delta_time.value, 0, 90, where=np.array(sunaltazs.alt.value < 0), color='0.5', zorder=0)
        plt.fill_between(delta_time.value, 0, 90, where=np.array(sunaltazs.alt.value < -18), color='k', zorder=0)
    elif "sunrise" in filename:
        plt.fill_between(delta_time.value, 0, 90, where=np.array(sunaltazs.alt.value < 0), color='0.5', zorder=0)
        plt.fill_between(delta_time.value, 0, 90, where=np.array(sunaltazs.alt.value < 18), color='k', zorder=0)


    plt.legend(loc='upper left')
    plt.xlim(-3, 3)
    plt.ylim(0, 90)
    plt.xlabel('Hours from MST')
    plt.ylabel('Altitude [deg]')
    plt.savefig(os.path.join('static/stargazing_plots', filename))
    plt.close()
    
def clear_old_images():
    today = datetime.date.today()
    for filename in os.listdir('static/stargazing_plots'):
        if filename.endswith('.png'):
            try:
                file_date_str = filename.split('_')[0]
                file_date = datetime.datetime.strptime(file_date_str, '%Y-%m-%d').date()
                if file_date < today:
                    print(f"Deleting {filename}") #Debugging print
                    os.remove(os.path.join('static/stargazing_plots', filename))
            except (ValueError, IndexError):
                pass

# Bulletin board data (using a simple text file for now)
BULLETIN_BOARD_FILE = "bulletin_board.txt"

# Create the file if it doesn't exist
if not os.path.exists(BULLETIN_BOARD_FILE):
    with open(BULLETIN_BOARD_FILE, "w") as f:
        f.write("")  # Start with an empty file
story_branches = ["start","win","investigate_drone","analyze_energy","contact_underwater",
    "follow_drone","stabilize_portal","evacuate","continue_contact","retreat_observe_underwater",
    "enter_city", "communicate_ship","prepare_encounter","help_them","decline_return",
    "accept_offer","decline_offer_ship"]
message_branches = ["investigate_drone","analyze_energy", "lockdown", "contact_underwater",
    "follow_drone", "stabilize_portal", "evacuate", "retreat_observe_underwater", "enter_city",
    "observe_city", "communicate_ship", "prepare_encounter", "help_them", "decline_return",
    "accept_offer", "decline_offer_ship", "win", "start", "run_coward", "return_flynn","watch_more"]

def load_game_data():
    try:
        with open(GAME_DATA_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"story_branches": {}}

def save_game_data(data):
    with open(GAME_DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)

def generate_css(drone, alien, underwater, experience, luck):
    """Generates wild CSS based on trust, experience, and luck."""
    css = f"""
    body {{
        background: linear-gradient(
            to top right, 
            hsl({drone * 3.6}, {drone}%, {drone * 0.5}%), 
            hsl({experience * 3.6 + 180}, {experience * 0.8}%, {experience * 0.4}%));
        color: hsl({alien * 3.6 + 30}, {alien * 0.6}%, {100 - alien * 0.5}%);
        border-color: hsl({drone * 3.6 + 240}, {drone * 0.7}%, {drone * 0.3}%);
        box-shadow: 0 0 {luck * 0.5}px rgba(255, 0, 0, {luck / 100}), 0 0 {luck * 0.3}px rgba(0, 255, 0, {luck / 100}), 0 0 {luck * 0.4}px rgba(0, 0, 255, {luck / 100});
        text-shadow: {alien * 0.2}px {alien * 0.2}px {alien * 0.1}px rgba(255, 215, 0, {alien / 100});
        font-weight: {200 + alien * 2};
        background: linear-gradient(
            to bottom, 
            hsl({underwater * 3.6 + 180}, {underwater * 0.8}%, {underwater * 0.3}%), 
            hsl({underwater * 3.6 + 300}, {underwater * 0.6}%, {underwater * 0.5}%));
        opacity: {1 - underwater * 0.01};
        filter: blur({underwater * 0.1}px);
        transform: rotate({luck * 0.1 - 5}deg) scale({1 + experience * 0.005});
        transition: all 0.5s ease;
    }}
    .container {{
        border: {experience * 0.1}px solid hsl({drone * 3.6 + 270}, {drone * 0.5}%, {drone * 0.2}%);
        border-radius: {luck * 0.2}px;
        transform: rotate({luck * 0.05 - 2.5}deg);
        box-shadow: 0 0 {experience * 0.2}px rgba(0, 0, 0, {experience / 100});
    }}
    .rad-title {{
        font-size: {2 + experience * 0.05}em;
        color: hsl({alien * 3.6 + 60}, {alien * 0.9}%, {90 - alien * 0.4}%);
    }}
    .rad-button {{
        background-color: hsl({luck * 3.6 + 120}, {luck * 0.7}%, {50 + luck * 0.2}%);
        color: hsl({experience * 3.6 + 300}, {experience * 0.9}%, {10 - experience * 0.1}%);
        border-radius: {luck * 0.1}px;
        transform: scale({1 + luck * 0.003});
    }}
    """
    return css
    
drone =  {}
alien  = {}
underwater  = {}
experience  = {}
luck  = {}



@app.route("/")
def index():
    return render_template("index.html")
    
@app.route("/flynnconvergence", methods=["GET", "POST"])
def home():
    global drone ,alien ,underwater ,experience,luck ,story_branches
    if "story" not in session:
        session["story"] = "start"
        session["style"] = "d0a0u0e0l0"
        session["inventory"] = []
        session["drone_knowledge"] = 0
        session["alien_trust"] = 0
        session["underwater_trust"] = 0
        session["user_experience"] = 0
        session["user_luck"] = random.randint(-3,3)

    story = session["story"]
    style = session["style"]
    message = ""
    choices = []
    inventory = session["inventory"]
    drone_knowledge = session["drone_knowledge"]
    alien_trust = session["alien_trust"]
    underwater_trust = session["underwater_trust"]
    user_experience = session["user_experience"]
    user_luck = session["user_luck"]
    game_data = load_game_data() #Load game data

    if request.method == "POST":
        choice = request.form["choice"]

        # Update trust levels *before* session update and rendering
        if choice == "start":
            session['underwater_trust'] = 0
            session['drone_knowledge'] = 0
            session['alien_trust'] = 0
            session['user_experience'] = 0
            session['user_luck'] = random.randint(1,100)
        else:
            session['underwater_trust'] = min(100, session['underwater_trust'] + underwater.get(choice, 0))
            session['drone_knowledge'] = min(100, session['drone_knowledge'] + drone.get(choice, 0))
            session['alien_trust'] = min(100, session['alien_trust'] + alien.get(choice, 0))
            session['user_experience'] = min(100, session['user_experience'] + experience.get(choice, 0))
            session['underwater_trust'] = max(0, session['underwater_trust'])
            session['drone_knowledge'] = max(0, session['drone_knowledge'] )
            session['alien_trust'] = max(0, session['alien_trust'])
            session['user_experience'] = max(0, session['user_experience'])
            session['user_luck'] =  luck.get(choice, 0)

        # Update session story and style
        session["story"] = choice
        session["style"] = f"d{session['drone_knowledge']}a{session['alien_trust']}u{session['underwater_trust']}e{session['user_experience']}l{session['user_luck']}"

    # Set message and choices *after* session update
    story = session["story"] #get story from the session.
    if story == "start":
        message = """You're a researcher at the remote Flynn Tower, monitoring unusual energy signatures. Suddenly, a shimmering drone materializes, emanating a strange hum.
        Do you:"""
        choices = [("Investigate the drone directly", "investigate_drone"), ("Analyze the energy readings from a safe distance", "analyze_energy"), ("Lockdown Flynn Tower", "lockdown")]
        drone["investigate_drone"] = 50
        alien["investigate_drone"] = 0
        underwater["investigate_drone"] = 0
        experience["investigate_drone"] = 10
        luck["investigate_drone"] = random.randint(1,100)
        drone["analyze_energy"] = 0
        alien["analyze_energy"] = 1
        underwater["analyze_energy"] = 0
        experience["analyze_energy"] = 1
        luck["analyze_energy"] = random.randint(1,100)
        drone["lockdown"] = 0
        alien["lockdown"] = 0
        underwater["lockdown"] = 0
        experience["lockdown"] = 1
        luck["lockdown"] = random.randint(1,100)
    elif story == "win":
        drone_knowledge = session['drone_knowledge']
        alien_trust = session['alien_trust']
        underwater_trust = session['underwater_trust']
        experiences = session['user_experience']
        lucky = session['user_luck']
        session["story"] = "start"
        #css_content = generate_css(int(drone_knowledge), int(alien_trust), int(underwater_trust), int(experiences), int(lucky))
        css_content = generate_css(0, 0, 0, 0, 50)
 
        return render_template("editor.html",css_content = css_content,story_branches=story_branches,message_branches=message_branches)
    elif story in game_data["story_branches"]:
        message = game_data["story_branches"][story]["message"]
        choices = game_data["story_branches"][story]["choices"]
        drone = game_data["story_branches"][story]["drone"]
        underwater = game_data["story_branches"][story]["underwater"]
        alien = game_data["story_branches"][story]["alien"]
        experience = game_data["story_branches"][story]["experience"]
        luck = game_data["story_branches"][story]["luck"]
    elif story == "investigate_drone":
        message = """As you approach the drone, it projects a holographic map, revealing an underwater city.
        Do you:"""
        choices = [
            ("Attempt to contact the underwater civilization", "contact_underwater"),
            ("Follow the drone's trajectory to the city", "follow_drone"),
        ]
        drone["follow_drone"] = 1
        alien["follow_drone"] = 0
        underwater["follow_drone"] = 0
        experience["follow_drone"] = 0
        luck["follow_drone"] = random.randint(1,100)
        drone["contact_underwater"] = 0
        alien["contact_underwater"] = 0
        underwater["contact_underwater"] = 1
        experience["contact_underwater"] = 1
        luck["contact_underwater"] = random.randint(1,100)

        
    elif story == "analyze_energy":
        message = """The energy readings are fluctuating wildly, indicating a potential interdimensional portal opening.
        Do you:"""
        choices = [
            ("Try to stabilize the portal", "stabilize_portal"),
            ("Alert the authorities and evacuate", "evacuate"),
        ]
        drone["stabilize_portal"] = 0
        alien["stabilize_portal"] = 1
        underwater["stabilize_portal"] = 0
        experience["stabilize_portal"] = 1
        luck["stabilize_portal"] = random.randint(1,100)
        drone["evacuate"] = 0
        alien["evacuate"] = 0
        underwater["evacuate"] = 0
        experience["evacuate"] = 1
        luck["evacuate"] = random.randint(1,100)
        
    elif story == "contact_underwater":
        message = """Your attempts to communicate are met with a series of complex sonic pulses. They seem to be a warning.
        Do you:"""
        choices = [
            ("Continue trying to communicate", "continue_contact"),
            ("Retreat and observe", "retreat_observe_underwater"),
            
        ]
        drone["retreat_observe_underwater"] = 0
        alien["retreat_observe_underwater"] = 0
        underwater["retreat_observe_underwater"] = -1
        experience["retreat_observe_underwater"] = 1
        luck["retreat_observe_underwater"] = random.randint(1,100)
        drone["continue_contact"] = 0
        alien["continue_contact"] = 0
        underwater["continue_contact"] = 1
        experience["continue_contact"] = 1
        luck["continue_contact"] = random.randint(1,100)
    elif story == "follow_drone":
        message = """You board a submersible and follow the drone. The journey is fraught with strange bioluminescent creatures. You arrive at a massive, glowing underwater city.
        Do you:"""
        choices = [
            ("Enter the city cautiously", "enter_city"),
            ("Observe from a distance", "observe_city"),
        ]
        drone["observe_city"] = 0
        alien["observe_city"] = 1
        underwater["observe_city"] = 0
        experience["observe_city"] = 1
        luck["observe_city"] = random.randint(1,100)
        drone["enter_city"] = 0
        alien["enter_city"] = 3
        underwater["enter_city"] = 5
        experience["enter_city"] = 1
        luck["enter_city"] = random.randint(1,100)
    elif story == "stabilize_portal":
        message = """Your efforts to stabilize the portal seem to be working, but it's attracting something... large. A massive extraterrestrial ship emerges.
        Do you:"""
        choices = [
            ("Attempt to communicate with the ship", "communicate_ship"),
            ("Prepare for a potential hostile encounter", "prepare_encounter"),
        ]
        drone["prepare_encounter"] = 0
        alien["prepare_encounter"] = 3
        underwater["prepare_encounter"] = 0
        experience["prepare_encounter"] = 1
        luck["prepare_encounter"] = random.randint(1,100)
        drone["communicate_ship"] = 0
        alien["communicate_ship"] = 1
        underwater["communicate_ship"] = 0
        experience["communicate_ship"] = 1
        luck["communicate_ship"] = random.randint(1,100)
        
    elif story == "evacuate":
        message = """You successfully evacuate the tower, but the portal destabilizes, causing a localized energy surge. The tower is destroyed. Game Over."""
        choices = [("Restart", "start")]
        drone["start"] = 0
        alien["start"] = 0
        underwater["start"] = 0
        experience["start"] = 0
        luck["start"] = random.randint(1,100)
    elif story == "continue_contact":
        message = """The sonic pulses intensify, then abruptly cease. The underwater city emits a powerful energy pulse. Just as if by magic is an alian is standing in front of you.
         Do you;"""
        choices = [("Tell the allian your name","tell_ailian_name"),("Run like the cowward you are", "run_coward")]
        drone["start"] = 0
        alien["start"] = 0
        underwater["start"] = 0
        experience["start"] = 0
        luck["start"] = random.randint(1,100)
    elif story == "retreat_observe_underwater":
        message = """You retreat and observe. The underwater city seems to be preparing for something. A large number of drones are being deployed."""
        choices = [("Watch More", "watch_more"),("Run like the cowward you are", "run_coward"),("Return to Flynn Tower and prepare like the hero you are", "return_flynn")]
        drone["start"] = 0
        alien["start"] = 0
        underwater["start"] = 0
        experience["start"] = 0
        luck["start"] = random.randint(1,100)
    elif story == "enter_city":
        message = """You enter the city. The inhabitants are welcoming, but their technology is far beyond human understanding. They reveal they are fleeing an approaching cosmic threat. They ask for your help.
        Do you:"""
        choices = [("Help them", "help_them"), ("Decline and return", "decline_return")]
        drone["help_them"] = 0
        alien["help_them"] = 0
        underwater["help_them"] = 3
        experience["help_them"] = 1
        luck["help_them"] = random.randint(1,100)
        drone["decline_return"] = 0
        alien["decline_return"] = 0
        underwater["decline_return"] = 0
        experience["decline_return"] = 0
        luck["decline_return"] = random.randint(1,100)
    elif story == "observe_city":
        message = """You observe the city from a distance. You see large numbers of drones exiting the city, heading towards the surface."""
        choices = [("Watch More", "watch_more"),("Run like the cowward you are", "run_coward"),("Return to Flynn Tower and prepare like the hero you are", "return_flynn")]
        drone["start"] = 0
        alien["start"] = 0
        underwater["start"] = 0
        experience["start"] = 0
        luck["start"] = random.randint(1,100)
    elif story == "communicate_ship":
        message = """You attempt to communicate. The ship responds with a complex series of light patterns. They seem to be explorers, but their intentions are unclear. They offer you passage to their home world.
        Do you:"""
        choices = [("Accept the offer", "accept_offer"), ("Decline the offer", "decline_offer_ship")]
    elif story == "prepare_encounter":
        message = """You prepare for a hostile encounter, but the ship simply scans the planet and departs. They were just passing through. Game Over."""
        choices = [("Restart", "start")]
        drone["start"] = 0
        alien["start"] = 0
        underwater["start"] = 0
        experience["start"] = 0
        luck["start"] = random.randint(1,100)
    elif story == "help_them":
        message = """You help the underwater civilization. Together, you devise a plan to shield the planet from the cosmic threat. You have saved the planet. You Win!"""
        choices = [("Restart", "start"),("Go to editor", "win")]
        drone["start"] = 0
        alien["start"] = 0
        underwater["start"] = 0
        experience["start"] = 0
        luck["start"] = random.randint(1,100)
        drone["win"] = 0
        alien["win"] = 0
        underwater["win"] = 0
        experience["win"] = 0
        luck["win"] = random.randint(1,100)      
    elif story == "decline_return":
        message = """You decline and return to the surface. You never forget the underwater city, but you never return. Game Over."""
        choices = [("Restart", "start")]
        drone["start"] = 0
        alien["start"] = 0
        underwater["start"] = 0
        experience["start"] = 0
        luck["start"] = random.randint(1,100)
    elif story == "accept_offer":
        message = """You accept the offer and board the ship. You embark on a journey to a distant star system. You Win!"""
        choices = [("Restart", "start"),("Go to editor", "win")]
        drone["start"] = 0
        alien["start"] = 0
        underwater["start"] = 0
        experience["start"] = 0
        luck["start"] = random.randint(1,100)
        drone["win"] = 0
        alien["win"] = 0
        underwater["win"] = 0
        experience["win"] = 0
        luck["win"] = random.randint(1,100)       
    elif story == "decline_offer_ship":
        message = """You decline the offer. The ship departs. You are left wondering what could have been. Game Over."""
        choices = [("Restart", "start")]
        drone["start"] = 0
        alien["start"] = 0
        underwater["start"] = 0
        experience["start"] = 0
        luck["start"] = random.randint(1,100)


    # Render the page
    drone_knowledge = session['drone_knowledge']
    alien_trust = session['alien_trust']
    underwater_trust = session['underwater_trust']
    experiences = session['user_experience']
    lucky = session['user_luck']
    css_content = generate_css(int(drone_knowledge), int(alien_trust), int(underwater_trust), int(experiences), int(lucky))
    return render_template("index_flynn_convergence.html", message=message, choices=choices, inventory=inventory, drone_knowledge=drone_knowledge, alien_trust=alien_trust, underwater_trust=underwater_trust, css_content=css_content)


@app.route("/api/save_branch", methods=["POST"])
def save_branch():
    global story_branches
    data = request.get_json()
    game_data = load_game_data()
    print(data["branch_id"][len(data["branch_id"])-7:] )
    if data["branch_id"][len(data["branch_id"])-7:] == '_fugawy':
        data["branch_id"] = data["branch_id"][:-7] 
    elif (data["branch_id"] in game_data["story_branches"]) or (data["branch_id"] in story_branches):
        return jsonify({"message": "Branch ID already exists"})
    game_data["story_branches"][data["branch_id"]] = {
        "message": data["message"],
        "choices": data["choices"],
        "drone": data["drone"],
        "underwater": data["underwater"],
        "alien": data["alien"],
        "experience": data["experience"],
        "luck": data["luck"],
    }
    save_game_data(game_data)
    return jsonify({"message": "Branch saved"})
    
def generate_planetarium_dome(latitude, longitude, observation_time,filename):
    """
    Generate a planetarium dome visualization using Matplotlib.
    
    :param latitude: Latitude of the observation location
    :param longitude: Longitude of the observation location
    :param observation_time: Datetime of observation
    :return: Base64 encoded image of the planetarium dome
    """
    # Set up observation location
    location = EarthLocation(lat=latitude*u.deg, lon=longitude*u.deg)
    
    # Convert observation time
    time = Time(observation_time)
    
    # Generate stars
    stars_data, sun_data, moon_data, planets_data = generate_star_catalog(location, time)
    
    # Create a new figure with increased figsize for legend
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='polar')
    
    # Plot stars
    for star_name, star_data in stars_data.items():
        ax.scatter(
            np.deg2rad(360 - star_data['azimuth']),  # Invert azimuth
            star_data['distance'],  # Distance from zenith
            c=star_data['color'],
            s=50 * star_data['size_factor'],  # Size based on magnitude
            alpha=0.8,
            edgecolors='white',
            label=star_name if star_data['is_bright'] else None
        )
    
    # Plot Sun if above horizon
    if sun_data['above_horizon']:
        ax.scatter(
            np.deg2rad(360 - sun_data['azimuth']),
            sun_data['distance'],
            c='yellow',
            s=300,
            alpha=1.0,
            edgecolors='orange',
            label='Sun'
        )
    
    # Plot Moon if above horizon
    if moon_data['above_horizon']:
        ax.scatter(
            np.deg2rad(360 - moon_data['azimuth']),
            moon_data['distance'],
            c='lightgray',
            s=200,
            alpha=1.0,
            edgecolors='gray',
            label='Moon'
        )
    
    # Plot planets
    for planet_name, planet_data in planets_data.items():
        if planet_data['above_horizon']:
            ax.scatter(
                np.deg2rad(360 - planet_data['azimuth']),
                planet_data['distance'],
                c=planet_data['color'],
                s=100,
                alpha=1.0,
                edgecolors='white',
                label=planet_name
            )
    
    # Create legend elements for star types
    star_types = {
        'blue': 'Hot, O/B type stars (>10,000K)',
        'white': 'A/F type stars (~7,500K)',
        'yellow': 'G type stars like our Sun (~5,800K)',
        'orange': 'K type stars (~4,500K)',
        'red': 'Cool, M type stars (<3,500K)'
    }
    
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=meaning,
                             markerfacecolor=color, markersize=10)
                       for color, meaning in star_types.items()]
    
    # Add planets to legend
    planet_legend = [
        Line2D([0], [0], marker='o', color='w', label='Mercury', markerfacecolor='gray', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Venus', markerfacecolor='yellow', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Mars', markerfacecolor='red', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Jupiter', markerfacecolor='sandybrown', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Saturn', markerfacecolor='gold', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Uranus', markerfacecolor='lightblue', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Neptune', markerfacecolor='blue', markersize=8)
    ]
    
    # Use ax.legend for the main stars/planets
    ax.legend(loc='best', bbox_to_anchor=(1.05, 1), title="Visible Objects")
    
    # Customize plot
    ax.set_theta_zero_location('N')  # North at the top
    ax.set_theta_direction(-1)  # Clockwise
    ax.set_ylim(90, 0)  # Zenith at center, horizon at edge
    ax.set_yticks(np.linspace(0, 90, 4))
    ax.set_yticklabels(['Zenith', '60°', '30°', 'Horizon'])
    
    # Add compass directions
    ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
    ax.set_xticklabels(['N', 'E', 'S', 'W'])
    
    plt.title(f'Planetarium Dome View\nLocation: {latitude}°N, {longitude}°E\nTime: {observation_time}')
    
    # Add a second legend for star types
    second_legend = fig.legend(handles=legend_elements, loc='lower right', 
                              bbox_to_anchor=(1, 0), title="Star Types")
    
    # Use explicit bbox_inches parameter for tight layout
    fig.tight_layout()
    
    # Save plot to a base64 encoded image
    #buf = io.BytesIO()
    fig.savefig(os.path.join('static/stargazing_plots', filename))
    #fig.savefig(buf, format='png', bbox_inches='tight')
    #buf.seek(0)
    #image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)  # Explicitly close the figure
    
    #return image_base64

def generate_star_catalog(location, time):
    """
    Generate a realistic star catalog including real stars, planets, sun and moon.
    
    :param location: EarthLocation object
    :param time: Observation time
    :return: Dictionaries containing star, sun, moon, and planet data
    """
    # Define bright stars with their real names and coordinates
    bright_stars = [
        ('Sirius', SkyCoord.from_name('Sirius')),
        ('Canopus', SkyCoord.from_name('Canopus')),
        ('Alpha Centauri', SkyCoord.from_name('Alpha Centauri')),
        ('Arcturus', SkyCoord.from_name('Arcturus')),
        ('Vega', SkyCoord.from_name('Vega')),
        ('Capella', SkyCoord.from_name('Capella')),
        ('Rigel', SkyCoord.from_name('Rigel')),
        ('Procyon', SkyCoord.from_name('Procyon')),
        ('Betelgeuse', SkyCoord.from_name('Betelgeuse')),
        ('Achernar', SkyCoord.from_name('Achernar')),
        ('Antares', SkyCoord.from_name('Antares')),
        ('Aldebaran', SkyCoord.from_name('Aldebaran')),
        ('Spica', SkyCoord.from_name('Spica')),
        ('Pollux', SkyCoord.from_name('Pollux')),
        ('Fomalhaut', SkyCoord.from_name('Fomalhaut')),
        ('Deneb', SkyCoord.from_name('Deneb')),
        ('Regulus', SkyCoord.from_name('Regulus')),
        ('Altair', SkyCoord.from_name('Altair'))
    ]
    
    # Create a frame for the specified time and location
    frame = AltAz(obstime=time, location=location)
    
    # Get Sun position
    sun = get_sun(time)
    sun_altaz = sun.transform_to(frame)
    
    # Get Moon position
    moon_altaz = get_body('moon', time).transform_to(frame)
    
    
    # Get planet positions
    planet_names = ['Mercury', 'Venus', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
    planet_colors = {
        'Mercury': 'gray',
        'Venus': 'yellow',
        'Mars': 'red',
        'Jupiter': 'sandybrown',
        'Saturn': 'gold',
        'Uranus': 'lightblue',
        'Neptune': 'blue'
    }
    
    # Dictionary to store stars data
    stars_data = {}
    
    # Process bright stars
    for name, star in bright_stars:
        star_altaz = star.transform_to(frame)
        # Only include if above horizon
        if star_altaz.alt.value > 0:
            # Star color based on spectral type (approximated)
            if name in ['Rigel', 'Spica', 'Regulus', 'Vega']:
                color = 'blue'
            elif name in ['Sirius', 'Fomalhaut', 'Altair', 'Procyon']:
                color = 'white'
            elif name in ['Capella', 'Alpha Centauri']:
                color = 'yellow'
            elif name in ['Aldebaran', 'Arcturus', 'Pollux']:
                color = 'orange'
            elif name in ['Betelgeuse', 'Antares']:
                color = 'red'
            else:
                color = 'white'
                
            stars_data[name] = {
                'azimuth': star_altaz.az.value,
                'altitude': star_altaz.alt.value,
                'distance': 90 - star_altaz.alt.value,
                'color': color,
                'size_factor': 5,  # Bright stars are larger
                'is_bright': True
            }
    
    # Add some dimmer stars (simulate with random generation)
    #num_dim_stars = 100
    
    # Generate random coordinates more densely along Milky Way plane (rough approximation)
    #ra = np.random.uniform(0, 360, num_dim_stars)
    # Bias towards galactic plane
    #dec = np.random.normal(0, 30, num_dim_stars)  # More stars near celestial equator
    
    # Convert to sky coordinates
    #dim_stars = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
    
    # Convert to Alt/Az coordinates
    #dim_stars_altaz = dim_stars.transform_to(frame)
    
    # Filter stars above horizon
    #visible_mask = dim_stars_altaz.alt > 0 * u.deg
    
    # Star colors based on realistic distribution
    #star_colors = ['blue', 'white', 'yellow', 'orange', 'red']
    # Real distribution roughly follows: 0.7% blue, 10% white, 13% yellow, 17% orange, 59.3% red
    #probabilities = [0.007, 0.1, 0.13, 0.17, 0.593]
    # Normalize probabilities
    #probabilities = [p/sum(probabilities) for p in probabilities]
    
    # Random but plausible magnitudes (brightness)
    #magnitudes = np.random.exponential(scale=2.0, size=sum(visible_mask)) + 1
    #magnitudes = np.clip(magnitudes, 1, 6)  # Clamp to reasonable range
    
    # Add dim stars to stars_data
    #for i in range(sum(visible_mask)):
    #    star_name = f"star_{i}"
    #    stars_data[star_name] = {
    #       'azimuth': dim_stars_altaz.az[visible_mask][i].value,
     #       'altitude': dim_stars_altaz.alt[visible_mask][i].value,
     #       'distance': 90 - dim_stars_altaz.alt[visible_mask][i].value,
     #       'color': np.random.choice(star_colors, p=probabilities),
    #        'size_factor': (7 - magnitudes[i]) / 2,  # Adjust size based on magnitude
    #        'is_bright': False
     #   }
    
    # Sun data
    sun_data = {
        'azimuth': sun_altaz.az.value,
        'altitude': sun_altaz.alt.value,
        'distance': 90 - sun_altaz.alt.value,
        'above_horizon': sun_altaz.alt.value > 0
    }
    
    # Moon data
    moon_data = {
        'azimuth': moon_altaz.az.value,
        'altitude': moon_altaz.alt.value,
        'distance': 90 - moon_altaz.alt.value,
        'above_horizon': moon_altaz.alt.value > 0
    }
    
    # Get planet positions
    planets_data = {}
    for planet_name in planet_names:
        try:
            planet = get_body(planet_name, time)
            planet_altaz = planet.transform_to(frame)
            
            planets_data[planet_name] = {
                'azimuth': planet_altaz.az.value,
                'altitude': planet_altaz.alt.value,
                'distance': 90 - planet_altaz.alt.value,
                'color': planet_colors[planet_name],
                'above_horizon': planet_altaz.alt.value > 0
            }
        except:
            # Skip if planet can't be found
            continue
    
    return stars_data, sun_data, moon_data, planets_data



    
@app.route("/garden")
def garden():
    image_dir = "static/images/garden"  # Path to your pictogram directory
    try:
        images = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))] 
    except FileNotFoundError:
        images = []  # Handle the case where the directory doesn't exist.
        print("garden directory not found.")
    return render_template("garden_journal.html", page_title="Garden Journal",images=images)
    
@app.route("/trails")
def trails():
    return render_template("trail_maps_and_info.html", page_title="Trail Maps & Info")
    
@app.route("/pictograms")
def pictograms():
    image_dir = "static/images/pictogram"  # Path to your pictogram directory
    try:
        images = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))] # list of image files.
    except FileNotFoundError:
        images = []  # Handle the case where the directory doesn't exist.
        print("Pictogram directory not found.")
    return render_template("pictogram_gallery.html", images=images)

    
@app.route("/flora_fauna")
def flora_fauna():
    image_dir = "static/images/flaura"
    try:
        images = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
    except FileNotFoundError:
        images = []
        print("flora directory not found.")
    return render_template("desert_flora_and_fauna_and_minerals.html", images=images)

@app.route("/image/<image_name>")
def display_image_caption(image_name):
    image_path = os.path.join("images/all-images", image_name)
    caption_file = os.path.join("static/images/all-images/captions", os.path.splitext(image_name)[0] + ".txt")
    caption = None

    try:
        with open(caption_file, "r") as f:
            caption = f.read()
            print(caption)
    except FileNotFoundError:
        print(f"Caption file not found for {image_name}")

    return render_template("image_caption.html", image_path=image_path, caption=caption, image_name=image_name)

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/stargazing')
def stargazing():
    today = datetime.date.today()
    date_str = today.strftime("%Y-%m-%d")
    midnight = Time(today.strftime('%Y-%m-%d 00:00:00')) - utcoffset

    # Calculate sunset and sunrise times (in UTC)
    times = midnight + np.linspace(-12, 12, 1000) * u.hour
    frame = AltAz(obstime=times, location=concho_bonito)
    sunaltazs = get_sun(times).transform_to(frame)

    # Find sunset and sunrise times
    sunset_time = None
    sunrise_time = None

    for i in range(len(sunaltazs.alt) - 1):
        if sunaltazs.alt[i].value > 0 and sunaltazs.alt[i + 1].value < 0:
            sunset_time = times[i] + (times[i + 1] - times[i]) * (sunaltazs.alt[i].value / (sunaltazs.alt[i].value - sunaltazs.alt[i + 1].value))
        elif sunaltazs.alt[i].value < 0 and sunaltazs.alt[i + 1].value > 0:
            sunrise_time = times[i] + (times[i + 1] - times[i]) * (-sunaltazs.alt[i].value / (sunaltazs.alt[i + 1].value - sunaltazs.alt[i].value))

    # Apply UTC offset to convert to local time (MST)
    if sunset_time:
        sunset_time = sunset_time + utcoffset
        sunset_str = sunset_time.strftime('%I:%M %p')
    else:
        sunset_str = "Sunset time not found"

    if sunrise_time:
        sunrise_time = sunrise_time + utcoffset
        sunrise_str = sunrise_time.strftime('%I:%M %p')
    else:
        sunrise_str = "Sunrise time not found"
        
    image_filenames = {}    
    planet_filenameR = f'{date_str}_sunrise_planet.png'
    CONCHO_BONITO_LAT = 34.543055555555554
    CONCHO_BONITO_LON = -109.58749999999999
    planet_filepath = os.path.join('static/stargazing_plots', planet_filenameR)
    if not os.path.exists(planet_filepath):
        generate_planetarium_dome(CONCHO_BONITO_LAT, CONCHO_BONITO_LON, sunrise_time,planet_filenameR)
    #image_filenames['sunrise']['planet'] = planet_filenameR
    planet_filename = f'{date_str}_sunset_planet.png'
    planet_filepath = os.path.join('static/stargazing_plots', planet_filename)
    if not os.path.exists(planet_filepath):
        generate_planetarium_dome(CONCHO_BONITO_LAT, CONCHO_BONITO_LON, sunset_time,planet_filename)
        
    directions = ['N', 'S', 'E', 'W']
    
    
    #image_filenames['sunset']['planet'] = planet_filename

    for time_key, time_value in {'sunset': sunset_time, 'sunrise': sunrise_time}.items():
        if time_value is not None:
            image_filenames[time_key] = {}
            for direction in directions:
                filename = f'{date_str}_{time_key}_{direction}.png'
                filepath = os.path.join('static/stargazing_plots', filename)
                if not os.path.exists(filepath):
                    generate_stargazing_plot(time_value, direction, filename)
                image_filenames[time_key][direction] = filename
        else:
            image_filenames[time_key] = {}
            for direction in directions:
                image_filenames[time_key][direction] = None

    return render_template('stargazing.html', image_filenames=image_filenames, date_str=date_str, sunset_str=sunset_str, sunrise_str=sunrise_str)

    
@app.route("/events")
def events():
    return render_template("local_events.html", page_title="Local Events")
    
@app.route("/passtest",methods=['GET','POST'])  
def passtest():
    #usern = ""
    #userpass = ""

    if request.method == "POST":
        #usern = request.form.get("username")
        #userpass = request.form.get("password")
        #if username in ('tom','james') and userpass in ('tom','james'):
        return render_template("kiva.html", page_title="Kiva")
    else:
        return render_template("passtest.html", page_title="passtest")
    
@app.route("/kiva",methods=['GET','POST']) 
def kiva_enter():
    usern = ''
    userpass = ''

    if request.method == "POST":
        usern = request.form.get("username")
        userpass = request.form.get("password")
        if usern in ('tom','james','Tom','James') and userpass in ('tom','james','Tom','James'):
            return render_template("kiva.html", page_title="Kiva")
        else:
            return render_template("passtest.html", page_title="passtest")
    else:
        return render_template("passtest.html", page_title="passtest")
    
@app.route('/weather')
def weather():
    headers = {'User-Agent': 'home_weather_page/1.0 (ttombbab@gmail.com)'}
    points_url = f"https://api.weather.gov/points/{CONCHO_BONITO_LAT},{CONCHO_BONITO_LON}"
    try:
        points_response = requests.get(points_url, headers=headers)
        points_response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        points_data = points_response.json()
        grid_x = points_data['properties']['gridX']
        grid_y = points_data['properties']['gridY']
        office = points_data['properties']['cwa']

        grid_url = f"https://api.weather.gov/gridpoints/{office}/{grid_x},{grid_y}/forecast"
        grid_response = requests.get(grid_url, headers=headers)
        grid_response.raise_for_status()
        forecast_data = grid_response.json()
        forecast_periods = forecast_data['properties']['periods']

        weather_data = []
        for period in forecast_periods[:7]: # get the next 7 periods.
            weather_data.append({
                'name': period['name'],
                'temperature': period['temperature'],
                'temperature_unit': period['temperatureUnit'],
                'short_forecast': period['shortForecast'],
                'detailed_forecast': period['detailedForecast'],
                'start_time': datetime.datetime.fromisoformat(period['startTime']).strftime("%I:%M %p"),
                'end_time': datetime.datetime.fromisoformat(period['endTime']).strftime("%I:%M %p"),
                'icon': period['icon'],
                'wind_speed': period.get('windSpeed', 'N/A'),
                'wind_direction': period.get('windDirection', 'N/A'),
            })
        today = datetime.date.today()
        date_str = today.strftime("%Y-%m-%d")
        return render_template('weather.html', weather_data=weather_data, date_str=date_str, page_title="Concho Bonito Weather")

    except requests.exceptions.RequestException as e:
        return f"Error fetching weather data: {e}"
    except (KeyError, ValueError) as e:
        return f"Error processing weather data: {e}"

@app.route("/llm",methods=['GET','POST'])  
def llm():
    generated_text = None
    selected_model = "tomchat-fugawy"

    if request.method == "POST":
        prompt = request.form.get("prompt")
        selected_model = request.form.get("model")
        if prompt:
            result = generate_text(prompt, model=selected_model)
            if result:
                try:
                    generated_text = result["response"]
                except KeyError:
                    generated_text = "Error: 'response' key not found in Ollama output."
                    print(result)
            else:
                generated_text = "Error communicating with Ollama server."

    return render_template("llm_interface.html", generated_text=generated_text, selected_model=selected_model)

@app.route("/bulletin", methods=["GET", "POST"])
def bulletin():
    messages = []
    try:
        with open(BULLETIN_BOARD_FILE, "r") as f:
            messages = f.readlines()
    except FileNotFoundError:
        pass  # Handle the case where the file doesn't exist (shouldn't happen now)

    if request.method == "POST":
        message = request.form.get("message")
        if message:
            with open(BULLETIN_BOARD_FILE, "a") as f:
                f.write(message + "\n")  # Add newline for each message
            return redirect(url_for("bulletin"))  # Redirect to refresh the page

    return render_template("bulletin_board.html", messages=messages)

@app.errorhandler(Exception)
def handle_error(e):
    return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    clear_old_images()
    app.run(host="0.0.0.0", port=5000, debug=True)
