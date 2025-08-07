export function parseCsvHeaders(fileContent: string): string[] {
    // Handle both \n and \r\n line endings by splitting on the first newline
    const firstLine = fileContent.split(/\r?\n/)[0];
  
    // This regex handles quoted and unquoted fields.
    const regex = /,(?=(?:[^""]*"[^""]*")*[^""]*$)/;
    
    if (!firstLine) return [];

    try {
        const headers = firstLine.split(regex).map(header => {
            // Remove quotes from start and end of the header
            return header.replace(/^"|"$/g, '').trim();
        });
        return headers.filter(h => h); // Filter out any empty headers
    } catch (error) {
        console.error("Error parsing CSV headers:", error);
        return [];
    }
}