import xml.etree.ElementTree as ET
import csv
import os

# Function to parse a single flow element
def parse_flow(flow):
    flow_data = {}
    flow_data['appName'] = flow.find('appName').text
    flow_data['totalSourceBytes'] = int(flow.find('totalSourceBytes').text)
    flow_data['totalDestinationBytes'] = int(flow.find('totalDestinationBytes').text)
    flow_data['totalDestinationPackets'] = int(flow.find('totalDestinationPackets').text)
    flow_data['totalSourcePackets'] = int(flow.find('totalSourcePackets').text)
    flow_data['direction'] = flow.find('direction').text
    flow_data['source'] = flow.find('source').text
    flow_data['protocolName'] = flow.find('protocolName').text
    flow_data['sourcePort'] = int(flow.find('sourcePort').text)
    flow_data['destination'] = flow.find('destination').text
    flow_data['destinationPort'] = int(flow.find('destinationPort').text)
    flow_data['startDateTime'] = flow.find('startDateTime').text
    flow_data['stopDateTime'] = flow.find('stopDateTime').text
    flow_data['Tag'] = flow.find('Tag').text
    return flow_data

# Use the current directory as the xml_folder
xml_folder = os.getcwd()

# Get a list of all XML files in the folder
xml_files = [f for f in os.listdir(xml_folder) if f.endswith('.xml')]

# Iterate through each XML file
for xml_file in xml_files:
    # Construct the full path to the XML file
    xml_file_path = os.path.join(xml_folder, xml_file)

    # Change the file extension from .xml to .csv
    csv_file_name = os.path.splitext(xml_file)[0] + '.csv'

    # Define the CSV file path for output
    csv_file_path = os.path.join(xml_folder, csv_file_name)

    # Initialize a list to store the extracted information
    results = []

    try:
        # Parse the XML data from the file
        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        # Iterate through all elements in the XML
        for element in root.iter():
            # Check if the element's name starts with "Testbed"
            if element.tag.startswith("Testbed"):
                try:
                    flow_data = parse_flow(element)
                    results.append(flow_data)
                except Exception as e:
                    print(f"Error parsing a flow element in '{xml_file}': {e}. Skipping this element.")


        # Write the extracted data to a CSV file
        with open(csv_file_path, mode='w', newline='') as csv_file:
            fieldnames = [
                'appName', 'totalSourceBytes', 'totalDestinationBytes',
                'totalDestinationPackets', 'totalSourcePackets', 'direction', 'source',
                'protocolName', 'sourcePort', 'destination', 'destinationPort',
                'startDateTime', 'stopDateTime', 'Tag'
            ]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            # Write the CSV header
            writer.writeheader()

            # Write the data for each flow
            for flow in results:
                writer.writerow(flow)

        print(f"Data from '{xml_file}' has been exported to '{csv_file_name}'")

    except ET.ParseError as e:
        print(f"Error parsing '{xml_file}': {e}. Skipping this file.")
        continue
