import './styles/tokens.css';
import './styles/main.css';
import { AppController } from './ui/appController.ts';

const root = document.getElementById('app');

if (root) {
  const controller = new AppController(root);
  void controller.init();

  if (typeof window !== 'undefined') {
    window.addEventListener('beforeunload', () => {
      controller.dispose();
    });
  }
}
