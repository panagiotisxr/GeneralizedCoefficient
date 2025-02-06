from math import sin, tan, radians

def calculate_pressure():
    # Access input values from the HTML
    H = float(Element("height").element.value)  # Total height of soil
    z = float(Element("depth").element.value)  # Depth
    c = float(Element("cohesion").element.value)  # Cohesion
    φ = float(Element("friction").element.value)  # Friction angle
    ah = float(Element("acceleration").element.value)  # Seismic acceleration

    # Check if the input is valid
    if z > H:
        pyscript.write("earth_pressure", "Error: Depth (z) cannot be greater than height (H).")
        return

    # Convert friction angle to radians
    φ_radians = radians(φ)

    # Calculate earth pressure using a simplified Rankine formula
    K_a = (1 - sin(φ_radians)) / (1 + sin(φ_radians))  # Active earth pressure coefficient
    pressure = K_a * (H - z) + 2 * c * tan(φ_radians) - ah * H  # Simplified formula

    # Display the result on the page
    pyscript.write("earth_pressure", round(pressure, 2))
