const exifr = require('exifr');

const analyzeMetadata = async (imagePath) => {
  try {
    const fs = require('fs').promises;
    const buffer = await fs.readFile(imagePath);
    const exif = await exifr.parse(buffer, {
      translateKeys: true,
      translateValues: true,
      tiff: true,
      xmp: true,
      icc: true,
      iptc: true,
      jfif: true
    });
    
    console.log('--- EXIF DEBUG INFO ---');
    console.log('Path:', imagePath);
    console.log('Buffer Size:', buffer.length);
    console.log('Data Found Keys:', exif ? Object.keys(exif).join(', ') : 'NONE');
    if (exif) {
      console.log('--- RAW EXIF OBJECT ---');
      console.log(JSON.stringify(exif));
      console.log('-----------------------');
    }
    console.log('-----------------------');

    let score = 50; // start neutral
    const flags = [];

    // No EXIF data at all → strong synthetic signal
    if (!exif) {
      return {
        metadata_score: 85,
        flags: ['No EXIF data found — strong synthetic signal'],
        raw: null
      };
    }

    // Check camera make/model
    if (!exif.Make && !exif.Model) {
      score += 20;
      flags.push('No camera make/model found');
    } else {
      score -= 20;
      flags.push(`Camera: ${exif.Make || ''} ${exif.Model || ''}`);
    }

    // Check software field
    if (exif.Software) {
      const aiSoftware = [
        'stable diffusion', 'midjourney', 'dall-e',
        'firefly', 'canva ai', 'runway', 'adobe firefly'
      ];
      const detectedSoftware = exif.Software.toLowerCase();
      const isAI = aiSoftware.some(ai => detectedSoftware.includes(ai));

      if (isAI) {
        score += 40;
        flags.push(`AI software detected: ${exif.Software}`);
      } else {
        flags.push(`Software: ${exif.Software}`);
      }
    } else {
      score += 10;
      flags.push('No software info found');
    }

    // Check timestamp
    if (!exif.DateTime) {
      score += 10;
      flags.push('No timestamp found');
    } else {
      score -= 10;
      flags.push(`Timestamp: ${exif.DateTime}`);
    }

    // Check GPS
    if (!exif.GPSLatitude && !exif.GPSLongitude) {
      score += 5;
      flags.push('No GPS data found');
    } else {
      score -= 10;
      flags.push('GPS data present');
    }

    // Clamp score between 0 and 100
    score = Math.min(100, Math.max(0, score));

    return {
      metadata_score: score,
      flags,
      raw: exif
    };

  } catch (err) {
    console.error('Metadata analysis error:', err.message);
    return {
      metadata_score: 50,
      flags: ['Could not parse metadata'],
      raw: null
    };
  }
};

module.exports = { analyzeMetadata };