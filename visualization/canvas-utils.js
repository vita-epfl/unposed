function drawPose(ctx, keypoints, isFuture, scale=1) {
    let segments = [
        [16, 14], [14, 12], [12, 6], [6, 8], [8, 10],
        [15, 13], [13, 11], [11, 5], [5, 7], [7, 9],
        [11, 12], [5, 6], [5, 0], [6, 0]
    ];
    for (let k = 0; k < segments.length; k++) {
        let [i, j] = segments[k];
        if (Math.min(keypoints[3*i+2], keypoints[3*j+2]) < LOW_CONFIDENCE_THRESHOLD) continue;

        color = ['#3B75AF', '#B3C6E5', '#A8DD93', '#8D69B8', '#84584E', '#EF8636', '#F5BE82', '#C53A32', '#C2B1D2', '#BE9E96', '#519E3E', '#F19D99', 'red', 'red'][k];
        if (isFuture) color = 'rgba(255, 255, 0, 0.8)';
        ctx.strokeStyle = color;
        ctx.lineWidth = 10 * scale;
        ctx.lineCap = 'round';
        ctx.beginPath();
        ctx.moveTo(keypoints[3*i], keypoints[3*i + 1]);
        ctx.lineTo(keypoints[3*j], keypoints[3*j + 1]);
        ctx.stroke();
    }

    for (let i = 0; i < keypoints.length; i += 3) {
        if (![12, 14, 16, 11, 13, 15, 0, 5, 7, 9, 6, 8, 10].includes(i/3)) continue;
        if (keypoints[i + 2] < LOW_CONFIDENCE_THRESHOLD) continue;

        const x = keypoints[i];
        const y = keypoints[i + 1];
        ctx.fillStyle = 'white';
        if (isFuture) ctx.fillStyle = 'black';
        ctx.beginPath();
        ctx.arc(x, y, 3 * scale, 0, 2 * Math.PI);
        ctx.fill();
    }

}

function drawArrow(ctx, startX, startY, dx, dy, r) {
    // draw circle
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(startX, startY, r + 5, 0, 2 * Math.PI);
    poseCtx.globalAlpha = 0.5;
    ctx.fillStyle = 'black';
    ctx.fill();
    poseCtx.globalAlpha = 1;
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(startX, startY); // Start at the given coordinates
    ctx.lineTo(startX + dx, startY + dy); // Draw line to the end point based on dx and dy
    ctx.stroke();

    // draw arrowhead
    ctx.beginPath();
    const headlen = 10;
    const angle = Math.atan2(dy, dx);
    ctx.moveTo(startX + dx, startY + dy);
    ctx.lineTo(startX + dx - headlen * Math.cos(angle - Math.PI / 6), startY + dy - headlen * Math.sin(angle - Math.PI / 6));
    ctx.lineTo(startX + dx - headlen * Math.cos(angle + Math.PI / 6), startY + dy - headlen * Math.sin(angle + Math.PI / 6));
    ctx.lineTo(startX + dx, startY + dy);
    ctx.fillStyle = 'white';
    ctx.fill();
}

function drawGrid(ctx, xMin, xMax, yMin, yMax) {
    const centerX = (xMin + xMax) / 2;
    const centerY = (yMin + yMax) / 2;
    const maxDistance = Math.sqrt((xMax - centerX) ** 2 + (yMax - centerY) ** 2);

    const numberOfLines = 8; // Number of grid lines

    const xSegmentLength = (xMax - xMin) / numberOfLines; // Horizontal segment length
    const ySegmentLength = (yMax - yMin) / numberOfLines; // Vertical segment length
    predCtx.lineCap = 'butt'; // or 'cap'


    // Function to calculate opacity based on distance from center
    function calculateOpacity(x, y) {
        const distance = Math.sqrt((x - centerX) ** 2 + (y - centerY) ** 2);
        return 1 - (distance / maxDistance);
    }

    // Draw vertical line segments
    for (let i = 1; i <= numberOfLines - 1; i++) {
        const x = xMin + i * ((xMax - xMin) / numberOfLines);
        for (let y = yMin; y < yMax; y += ySegmentLength) {
            const opacity = calculateOpacity(x, y + ySegmentLength / 2);
            ctx.strokeStyle = `rgba(255, 255, 255, ${opacity.toFixed(2)})`;
            ctx.beginPath();
            ctx.moveTo(x, y);
            ctx.lineTo(x, y + ySegmentLength);
            ctx.stroke();
        }
    }

    // Draw horizontal line segments
    for (let j = 1; j <= numberOfLines - 1; j++) {
        const y = yMin + j * ((yMax - yMin) / numberOfLines);
        for (let x = xMin; x < xMax; x += xSegmentLength) {
            const opacity = calculateOpacity(x + xSegmentLength / 2, y);
            ctx.strokeStyle = `rgba(255, 255, 255, ${opacity.toFixed(2)})`;
            ctx.beginPath();
            ctx.moveTo(x, y);
            ctx.lineTo(x + xSegmentLength, y);
            ctx.stroke();
        }
    }
}
