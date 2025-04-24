import obd
import time
import datetime
import csv
import serial
import pynmea2
import threading
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Configuració del port sèrie OBD-II (canvia-ho si cal)
port_obd = 'COM6'
baudrate_obd = 115200

# Configuració del port sèrie del GPS (canvia-ho si cal)
port_gps = 'COM7'
baudrate_gps = 9600

# Variables globals per emmagatzemar les últimes coordenades GPS i la història
ultima_latitud_gps = None
ultima_longitud_gps = None
historial_latituds = []
historial_longituds = []

# Esdeveniment per indicar quan s'han rebut noves coordenades GPS
noves_coordenades_event = threading.Event()

# Comandes OBD-II per a les dades que volem
commands = {
    "velocitat": obd.commands.SPEED,
    "rpm": obd.commands.RPM
}

# Generar el nom base del fitxer CSV i de la imatge
ara = datetime.datetime.now()
nom_base = f"dades_cotxe_gps_ruta_realtime_{ara.strftime('%Y%m%d_%H%M%S')}"
nom_fitxer_csv = f"{nom_base}.csv"
nom_imatge_ruta = f"{nom_base}_ruta.png"

# Freqüència de mostreig en segons
freq_mostreig = 0.5

# Funció per obtenir la data i hora actuals
def obtenir_data_hora_actual():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def llegir_coordenades_gps():
    global ultima_latitud_gps
    global ultima_longitud_gps
    global historial_latituds
    global historial_longituds
    try:
        with serial.Serial(port_gps, baudrate_gps, timeout=1) as ser_gps:
            print(f"Fil GPS (Temps Real): Connectat al GPS al port {port_gps} amb baudrate {baudrate_gps}")
            print("Fil GPS (Temps Real): Esperant dades GPS...")
            while True:
                line = ser_gps.readline().decode('utf-8', errors='ignore').strip()
                if '$GPGGA' in line or '$GPRMC' in line:
                    try:
                        msg = pynmea2.parse(line)
                        if hasattr(msg, 'latitude') and hasattr(msg, 'longitude'):
                            ultima_latitud_gps = msg.latitude
                            ultima_longitud_gps = msg.longitude
                            historial_latituds.append(ultima_latitud_gps)
                            historial_longituds.append(ultima_longitud_gps)
                            noves_coordenades_event.set()
                            # print(f"Fil GPS (Temps Real): Latitud: {ultima_latitud_gps:.6f}, Longitud: {ultima_longitud_gps:.6f}")
                    except pynmea2.ParseError as e:
                        print(f"Fil GPS (Temps Real): Error al parsejar la frase NMEA: {line} - Error: {e}")
                time.sleep(0.1)
    except serial.SerialException as e:
        print(f"Fil GPS (Temps Real): Error al connectar amb el GPS al port {port_gps}: {e}")
    except KeyboardInterrupt:
        print("Fil GPS (Temps Real): Finalitzat per l'usuari.")
    finally:
        if 'ser_gps' in locals() and ser_gps.is_open:
            ser_gps.close()
            print("Fil GPS (Temps Real): Connexió GPS tancada.")

# Funció principal per llegir les dades OBD-II i GPS amb visualització en temps real
def llegir_dades_obd_gps_realtime():
    global ultima_latitud_gps
    global ultima_longitud_gps
    global historial_latituds
    global historial_longituds
    global fig

    # Inicia el fil per a la lectura del GPS
    fil_gps = threading.Thread(target=llegir_coordenades_gps)
    fil_gps.daemon = True
    fil_gps.start()

    connection = None
    try:
        connection = obd.OBD(port_obd, baudrate=baudrate_obd)
        if connection.is_connected():
            print(f"Fil Principal (Temps Real): Connectat a l'adaptador OBD-II al port {port_obd}")
            print(f"Fil Principal (Temps Real): Les dades es guardaran a l'arxiu: {nom_fitxer_csv}")

            # Escriure la capçalera al fitxer CSV
            with open(nom_fitxer_csv, 'w', newline='') as csvfile:
                camp_noms = ['Data Hora', 'Velocitat (km/h)', 'RPM', 'Latitud', 'Longitud']
                writer = csv.DictWriter(csvfile, fieldnames=camp_noms)
                writer.writeheader()

            # Inicialitzar la figura per al gràfic
            fig, ax = plt.subplots(figsize=(10, 8))
            line, = ax.plot([], [], marker='o', linestyle='-', linewidth=1, markersize=3)
            ax.set_xlabel("Longitud")
            ax.set_ylabel("Latitud")
            ax.set_title("Ruta Recorreguda en Temps Real")
            ax.grid(True)

            def update_plot(frame):
                line.set_data(historial_longituds, historial_latituds)
                # Ajustar els límits del gràfic dinàmicament (opcional)
                if historial_latituds and historial_longituds:
                    min_lat, max_lat = min(historial_latituds), max(historial_latituds)
                    min_lon, max_lon = min(historial_longituds), max(historial_longituds)
                    padding_lat = (max_lat - min_lat) * 0.1 if (max_lat - min_lat) > 0 else 0.01
                    padding_lon = (max_lon - min_lon) * 0.1 if (max_lon - min_lon) > 0 else 0.01
                    ax.set_xlim(min_lon - padding_lon, max_lon + padding_lon)
                    ax.set_ylim(min_lat - padding_lat, max_lat + padding_lat)
                return line,

            ani = FuncAnimation(fig, update_plot, interval=1000, blit=True)

            while True:
                dades = {}
                dades['Data Hora'] = obtenir_data_hora_actual()

                # Obtenir la velocitat
                response_speed = connection.query(commands["velocitat"])
                if not response_speed.is_null():
                    dades['Velocitat (km/h)'] = response_speed.value.to("km/h").magnitude
                else:
                    dades['Velocitat (km/h)'] = None

                # Obtenir les RPM
                response_rpm = connection.query(commands["rpm"])
                if not response_rpm.is_null():
                    dades['RPM'] = response_rpm.value.magnitude
                else:
                    dades['RPM'] = None

                # Obtenir les últimes coordenades GPS vàlides
                dades['Latitud'] = ultima_latitud_gps
                dades['Longitud'] = ultima_longitud_gps

                # Mostrar les dades per pantalla
                print(f"Fil Principal (Temps Real): Data Hora: {dades['Data Hora']}, Velocitat: {dades.get('Velocitat (km/h)', 'N/A')} km/h, RPM: {dades.get('RPM', 'N/A')}, Latitud: {dades.get('Latitud', 'N/A')}, Longitud: {dades.get('Longitud', 'N/A')}")

                # Guardar les dades al fitxer CSV
                with open(nom_fitxer_csv, 'a', newline='') as csvfile:
                    camp_noms = ['Data Hora', 'Velocitat (km/h)', 'RPM', 'Latitud', 'Longitud']
                    writer = csv.DictWriter(csvfile, fieldnames=camp_noms)
                    writer.writerow(dades)

                time.sleep(freq_mostreig)

            plt.show()

        else:
            print(f"Fil Principal (Temps Real): No s'ha pogut connectar a l'adaptador OBD-II al port {port_obd}. Verifica la connexió i el port.")

    except obd.exceptions.OBDError as e:
        print(f"Fil Principal (Temps Real): Error de connexió OBD-II: {e}")
    except serial.SerialException as e:
        print(f"Fil Principal (Temps Real): Error de port sèrie (OBD): {e}. Assegura't que el port '{port_obd}' és correcte i que tens els permisos necessaris.")
    except KeyboardInterrupt:
        print("Fil Principal (Temps Real): Programa finalitzat per l'usuari. Guardant la ruta...")
        if 'fig' in locals():
            plt.savefig(nom_imatge_ruta)
            print(f"Fil Principal (Temps Real): Ruta guardada com a {nom_imatge_ruta}")
    except Exception as e:
        print(f"Fil Principal (Temps Real): S'ha produït un error inesperat: {e}")
    finally:
        if connection and connection.is_connected():
            connection.close()
            print("Fil Principal (Temps Real): Connexió OBD-II tancada.")

if __name__ == "__main__":
    llegir_dades_obd_gps_realtime()