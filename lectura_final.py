import obd
import time
import datetime
import csv
import serial
import pynmea2
import threading

# Configuració del port sèrie OBD-II (canvia-ho si cal)
port_obd = 'COM6'
baudrate_obd = 115200

# Configuració del port sèrie del GPS (canvia-ho si cal)
port_gps = 'COM7'
baudrate_gps = 9600

# Variables globals per emmagatzemar les últimes coordenades GPS
ultima_latitud_gps = None
ultima_longitud_gps = None

# Esdeveniment per indicar quan s'han rebut noves coordenades GPS (no s'utilitza activament aquí)
noves_coordenades_event = threading.Event()

# Comandes OBD-II per a les dades que volem
commands = {
    "velocitat": obd.commands.SPEED,
    "rpm": obd.commands.RPM
}

# Generar el nom base del fitxer CSV
ara = datetime.datetime.now()
nom_base = f"dades_cotxe_gps_sense_grafic_{ara.strftime('%Y%m%d_%H%M%S')}"
nom_fitxer_csv = f"{nom_base}.csv"

# Freqüència de mostreig en segons
freq_mostreig = 0.5

# Funció per obtenir la data i hora actuals
def obtenir_data_hora_actual():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def llegir_coordenades_gps():
    global ultima_latitud_gps
    global ultima_longitud_gps
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
                            noves_coordenades_event.set()
                            # print(f"Fil GPS (Temps Real): Latitud: {ultima_latitud_gps:.6f}, Longitud: {ultima_longitud_gps:.6f}")
                    except pynmea2.ParseError as e:
                        print(f"Fil GPS (Temps Real): Error al parsejar la frase NMEA: {line} - Error: {e}")
                time.sleep(0.1)
    except serial.SerialException as e:
        print(f"Fil GPS (Temps Real): Error al connectar amb el GPS al port {port_gps}: {e}")
    except KeyboardInterrupt:
        print("Fil GPS (Temps Real): Finalitzat per l'usuari (fil GPS).")
    finally:
        if 'ser_gps' in locals() and ser_gps.is_open:
            ser_gps.close()
            print("Fil GPS (Temps Real): Connexió GPS tancada.")

# Funció principal per llegir les dades OBD-II i GPS sense visualització
def llegir_dades_obd_gps():
    global ultima_latitud_gps
    global ultima_longitud_gps

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

        else:
            print(f"Fil Principal (Temps Real): No s'ha pogut connectar a l'adaptador OBD-II al port {port_obd}. Verifica la connexió i el port.")

    except obd.exceptions.OBDError as e:
        print(f"Fil Principal (Temps Real): Error de connexió OBD-II: {e}")
    except serial.SerialException as e:
        print(f"Fil Principal (Temps Real): Error de port sèrie (OBD): {e}. Assegura't que el port '{port_obd}' és correcte i que tens els permisos necessaris.")
    except KeyboardInterrupt:
        print("Fil Principal (Temps Real): Programa finalitzat per l'usuari.")
    except Exception as e:
        print(f"Fil Principal (Temps Real): S'ha produït un error inesperat: {e}")
    finally:
        if connection and connection.is_connected():
            connection.close()
            print("Fil Principal (Temps Real): Connexió OBD-II tancada.")

if __name__ == "__main__":
    llegir_dades_obd_gps()