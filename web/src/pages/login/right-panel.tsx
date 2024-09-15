import { Flex, Typography } from 'antd';
import classNames from 'classnames';

import { useTranslate } from '@/hooks/common-hooks';
import styles from './index.less';

const { Title, Text } = Typography;

const LoginRightPanel = () => {
  const { t } = useTranslate('login');
  return (
    <section className={styles.rightPanel}>
      <Flex vertical gap={40}>
        <Title
          level={1}
          className={classNames(styles.white, styles.loginTitle)}
        >
          {t('title')}
        </Title>
      </Flex>
    </section>
  );
};

export default LoginRightPanel;
