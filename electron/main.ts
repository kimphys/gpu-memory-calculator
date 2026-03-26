import { app, BrowserWindow, protocol } from 'electron';
import * as path from 'path';
import * as fs from 'fs';

let mainWindow: BrowserWindow | null = null;

// Register custom protocol as privileged
protocol.registerSchemesAsPrivileged([
  { scheme: 'app', privileges: { secure: true, standard: true, supportFetchAPI: true, allowServiceWorkers: true } }
]);

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js'),
    },
    title: "LLM GPU Simulator",
    backgroundColor: '#0f1115',
    show: false,
  });

  const isDev = !app.isPackaged;
  
  if (!isDev) {
    // Register custom protocol for production
    protocol.handle('app', async (request) => {
      try {
        let pathname = new URL(request.url).pathname;
        if (pathname.startsWith('/')) pathname = pathname.slice(1);
        
        const filePath = path.join(__dirname, '../../out', pathname || 'index.html');
        
        const extension = path.extname(filePath).toLowerCase();
        let contentType = 'text/html';
        if (extension === '.css') contentType = 'text/css';
        else if (extension === '.js' || extension === '.mjs') contentType = 'text/javascript';
        else if (extension === '.svg') contentType = 'image/svg+xml';
        else if (extension === '.png') contentType = 'image/png';
        else if (extension === '.jpg' || extension === '.jpeg') contentType = 'image/jpeg';
        else if (extension === '.json') contentType = 'application/json';
        else if (extension === '.woff2') contentType = 'font/woff2';

        const data = await fs.promises.readFile(filePath);
        return new Response(data, {
          headers: { 'Content-Type': contentType }
        });
      } catch (e) {
        console.error('Failed to load resource:', e);
        return new Response('Not Found', { status: 404 });
      }
    });
  }

  const startUrl = isDev 
    ? 'http://localhost:3000' 
    : 'app://index.html';

  console.log('App starting...', { isDev, startUrl });
  mainWindow.loadURL(startUrl);

  mainWindow.once('ready-to-show', () => {
    mainWindow?.show();
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
  });

  if (isDev) {
    // mainWindow.webContents.openDevTools();
  }
}

app.on('ready', createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (mainWindow === null) {
    createWindow();
  }
});
